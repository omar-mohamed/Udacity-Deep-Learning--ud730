from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data


text = read_data(filename)
print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
bigram_vocabulary_size = vocabulary_size * vocabulary_size
first_letter = ord(string.ascii_lowercase[0])


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(int(dictid) + first_letter - 1)
    else:
        return ' '


batch_size = 64
num_unrollings = 10


class BatchGeneratorBigram(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size_actual = len(text)
        self._text_size_bigrams = self._text_size_actual // 2
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size_bigrams // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=self._batch_size, dtype=np.int)
        for b in range(self._batch_size):
            first_char_idx = self._cursor[b] * 2
            second_char_index = first_char_idx + 1
            batch[b] = char2id(self._text[first_char_idx]) * vocabulary_size
            if second_char_index != self._text_size_actual:
                batch[b] += char2id(self._text[second_char_index])

            self._cursor[b] = (self._cursor[b] + 1) % self._text_size_bigrams
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def bigramCharacters(encoding):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return id2char(encoding // vocabulary_size) + id2char(encoding % vocabulary_size)


def bigrams(encodings):
    return [bigramCharacters(bigram) for bigram in encodings]


def bigrams2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, bigrams(b))]
    return s


def bi_batch_one_hot(encodings, size=bigram_vocabulary_size):
    one_hot = np.zeros(shape=[len(encodings), size], dtype=np.float)
    index = 0
    for e in encodings:
        one_hot[index] = np.zeros(size)
        one_hot[index, e] = 1
        index = index + 1
    return one_hot


train_batches = BatchGeneratorBigram(train_text, batch_size, num_unrollings)
valid_batches = BatchGeneratorBigram(valid_text, 1, 1)

print(bigrams2string(train_batches.next()))
print(bigrams2string(train_batches.next()))
print(bigrams2string(valid_batches.next()))
print(bigrams2string(valid_batches.next()))


def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction, size=bigram_vocabulary_size):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution(size=bigram_vocabulary_size):
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, size])
    return b / np.sum(b, 1)[:, None]


num_nodes = 64
embedding_size = 128
graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([embedding_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, bigram_vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([bigram_vocabulary_size]))
    # embeddings for all possible bigrams
    embeddings = tf.Variable(tf.random_uniform([bigram_vocabulary_size, embedding_size], -1.0, 1.0))
    # one hot encoding for labels in
    one_hot_matrix = bi_batch_one_hot(list(range(bigram_vocabulary_size)), bigram_vocabulary_size)
    labels_one_hot = tf.constant(np.reshape(one_hot_matrix, -1), dtype=tf.float32,
                                 shape=[bigram_vocabulary_size, bigram_vocabulary_size])

    dropout_keep_prob = tf.placeholder(tf.float32)


    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        i = tf.nn.dropout(i, dropout_keep_prob)
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        output = output_gate * tf.tanh(state)
        output = tf.nn.dropout(output, dropout_keep_prob)
        return output, state


    # Input data.
    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(
            tf.placeholder(tf.int32, shape=[batch_size]))
    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.
    train_labels = list()
    for l in train_data[1:]:
        train_labels.append(tf.gather(labels_one_hot, l))

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell(tf.nn.embedding_lookup(embeddings, i), output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),
                                  saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.concat(train_labels, 0), logits=logits))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.int32, shape=[1])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(
        tf.nn.embedding_lookup(embeddings, sample_input), saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        feed_dict[dropout_keep_prob] = 0.75
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = list(batches)[1:]
            labels = np.concatenate([bi_batch_one_hot(l) for l in labels])
            print('Minibatch perplexity: %.2f' % float(
                np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed = np.argmax(sample(random_distribution(bigram_vocabulary_size), bigram_vocabulary_size))
                    sentence = bigramCharacters(feed)
                    reset_sample_state.run()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input: [feed], dropout_keep_prob: 1.0})
                        feed = np.argmax(sample(prediction, bigram_vocabulary_size))
                        sentence += bigramCharacters(feed)
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 0
            for _ in range(valid_size):
                b = valid_batches.next()
                predictions = sample_prediction.eval({sample_input: b[0], dropout_keep_prob: 1.0})
                # print(predictions)
                valid_logprob = valid_logprob + logprob(predictions, bi_batch_one_hot(b[1], bigram_vocabulary_size))
            print('Validation set perplexity: %.2f' % float(np.exp(
                valid_logprob / valid_size)))
