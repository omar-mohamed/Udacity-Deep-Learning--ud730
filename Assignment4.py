from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1  # grayscale


def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


batch_size = 64
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    conv1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth],
        stddev=0.1))  # y_size_filter,x_size_filter, input_depth, out_depth
    conv1_biases = tf.Variable(tf.zeros([depth]))
    conv2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    conv2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    # conv3_weights = tf.Variable(tf.truncated_normal(
    #     [patch_size, patch_size, depth, depth], stddev=0.1))
    # conv3_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    hidden1_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    hidden1_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    hidden2_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_hidden], stddev=0.1))
    hidden2_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    hidden3_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    hidden3_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data, use_dropout=False):
        conv = tf.nn.conv2d(data, conv1_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(value=conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv =  tf.nn.local_response_normalization(conv);
        hidden = tf.nn.relu(conv + conv1_biases)

        conv = tf.nn.conv2d(hidden, conv2_weights, [1, 1, 1, 1], padding='SAME')
       # conv = tf.nn.max_pool(value=conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.local_response_normalization(conv);
        hidden = tf.nn.relu(conv + conv2_biases)

        conv = tf.nn.conv2d(hidden, conv2_weights, [1, 1, 1, 1], padding='SAME')
 #       conv = tf.nn.local_response_normalization(conv);
        conv = tf.nn.max_pool(value=conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + conv2_biases)


        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, hidden1_weights) + hidden1_biases)
        if use_dropout:
            hidden = tf.nn.dropout(hidden, 0.75)
        hidden = tf.nn.relu(tf.matmul(hidden, hidden2_weights) + hidden2_biases)
        if use_dropout:
            hidden = tf.nn.dropout(hidden, 0.75)
        return tf.matmul(hidden, hidden3_weights) + hidden3_biases


    # Training computation.
    logits = model(tf_train_dataset, True)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
    # decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.05
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               500, 0.96, staircase=False)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    with tf.device('/cpu:0'):
        test_prediction = tf.nn.softmax(model(tf_test_dataset))


num_steps = 4001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions,lr = session.run(
            [optimizer, loss, train_prediction,learning_rate], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Learning rate at step %d: %f' % (step, lr))
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            # print('Validation accuracy: %.1f%%' % accuracy(
            #     valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
