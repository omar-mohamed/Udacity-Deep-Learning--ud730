# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

all_data = pickle.load(open('notMNIST.pickle', 'rb'))

image_size = 28
num_labels = 10

train_dataset = all_data['train_dataset']
train_labels = all_data['train_labels']
valid_dataset = all_data['valid_dataset']
valid_labels = all_data['valid_labels']
test_dataset = all_data['test_dataset']
test_labels = all_data['test_labels']

del all_data


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.


batch_size = 1024

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    hidden_layer_size = 128
    hidden_layer2_size = 256
    # Variables.
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, hidden_layer_size]))
    biases = tf.Variable(tf.zeros([hidden_layer_size]))

    # Training computation.
    logits = tf.add(tf.matmul(tf_train_dataset, weights), biases)

    relu_output = tf.nn.relu(logits)

    weights2 = tf.Variable(
        tf.truncated_normal([hidden_layer_size, hidden_layer2_size]))
    biases2 = tf.Variable(tf.zeros([hidden_layer2_size]))

    # Training computation.
    logits2 = tf.add(tf.matmul(relu_output, weights2), biases2)

    relu_output2 = tf.nn.relu(logits2)

    # relu_output=tf.nn.dropout(relu_output,0.5) #drop out regularization

    weights3 = tf.Variable(
        tf.truncated_normal([hidden_layer2_size, num_labels]))  # output layer weights
    biases3 = tf.Variable(tf.zeros([num_labels]))

    logits3 = tf.add(tf.matmul(relu_output2, weights3), biases3)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits3))

    regulirazation_lambda = 0.01
    loss = loss + regulirazation_lambda * tf.nn.l2_loss(weights) + regulirazation_lambda * tf.nn.l2_loss(
        weights2) + regulirazation_lambda * tf.nn.l2_loss(weights3)  # l2-regularization

    # Optimizer.

    # tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
    # decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.96, staircase=False)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits3)
    valid_prediction_output = tf.matmul(
        tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights) + biases), weights2) + biases2),
        weights3) + biases3
    valid_prediction = tf.nn.softmax(valid_prediction_output)
    test_prediction_output = tf.matmul(
        tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights) + biases), weights2) + biases2),
        weights3) + biases3
    test_prediction = tf.nn.softmax(test_prediction_output)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 10 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
