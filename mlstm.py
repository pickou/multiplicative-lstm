# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
import codecs
import pickle
from MultiplicativeLSTMCell import MultiplicativeLSTMCell


def preprocess_sms(filename):
    data = pd.read_csv(filename, names=['comment'],
                        usecols=[2], delimiter='\t')
    #print(data.comment.tolist())
    str_list = data.comment.tolist()
    for i in range(len(str_list)):
        str_list[i] = str(str_list[i]).decode('utf-8')
    text = "\n".join(str_list)
    #print(text)
    vocab = set(text)
    vocab_to_int = {c: i for i,c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

    return vocab, vocab_to_int, int_to_vocab, encoded

#input 
# example for one txt
def preprocess(filename, encoding='utf-8', saveVocab=False):
    f = codecs.open(filename, 'r', encoding=encoding)
    text = f.read()
    
    if os.path.exists('vocab.pkl'):
        with open('vocab.pkl', 'rb') as fr:
            vocab = pickle.load(fr)
    else:
        vocab = list(set(text))
    
    if saveVocab:
        with open('vocab.pkl', 'wb') as fw:
            pickle.dump(vocab, fw, protocol=2)

    vocab_to_int = {c: i for i,c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
    
    return vocab, vocab_to_int, int_to_vocab, encoded


# batch
def gen_batches(arr, n_seqs, n_steps):
    """
    arr: input array
    n_seqs: batch count 
    n_steps: sequence length 
    """
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr)/batch_size)

    # to integer
    arr = arr[: batch_size * n_batches]

    # reshape
    arr = np.reshape(arr, (n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        y[:, :-1], y[:,-1] = x[:, 1:], y[:, 0]
        yield x, y


# inputs prams
def build_inputs(batch_size, num_steps):
    inputs = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob
    
# single layer multiplicative lstm
def create_mlstm(hidden_units, batch_size, keep_prob):    
    lstm_cell = MultiplicativeLSTMCell(hidden_units, forget_bias=1.0, state_is_tuple=False)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=keep_prob)
    # single layer
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell])
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    return lstm_cell, init_state

# output with softmax
def build_outputs(lstm_output, in_size, out_size):
    seq_out = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_out, [-1, in_size])
    
    with tf.variable_scope('softmax'):
        w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        b = tf.Variable(tf.zeros(out_size))
    
    logits = tf.matmul(x, w) + b
    out = tf.nn.softmax(logits, name='predict')

    return out, logits

# loss 
def build_loss(logits, targets, hidden_units, num_class):
    y_one_hot = tf.one_hot(targets, num_class)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

# optimizer
def build_optimizer(loss, lr, grad_clip):
    tvars = tf.trainable_variables()
    # solve the gradient explosion
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(lr)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer


# model
class bytemLSTM:
    def __init__(self, num_class, batch_size=64, num_steps=100,
                hidden_units=512, lr=0.001, grad_clip=5, sampling=False):
        if sampling == True:
            self.batch_size, self.num_steps = 1, 1
        else:
            self.batch_size, self.num_steps = batch_size, num_steps
        
        tf.reset_default_graph()
        self.inputs, self.targets, self.keep_prob = build_inputs(self.batch_size, self.num_steps)
        x_one_hot = tf.one_hot(self.inputs, num_class)
        cell, self.init_state = create_mlstm(hidden_units, self.batch_size, self.keep_prob)
        self.outputs, self.state = tf.nn.dynamic_rnn(cell, x_one_hot, dtype=tf.float32, 
                                                     initial_state=self.init_state)
        self.pred, self.logits = build_outputs(self.outputs, hidden_units, num_class)
        self.loss = build_loss(self.logits, self.targets, hidden_units, num_class)
        self.optimizer = build_optimizer(self.loss, lr, grad_clip)


def Train(vocab, encoded, epochs, n_save, batch_size, 
          num_steps, keep_prob):
    model = bytemLSTM(len(vocab))
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        counter = 0
        new_state = sess.run(model.init_state)
        for e in range(epochs):
            loss = 0
            for x, y in gen_batches(encoded, batch_size, num_steps):
                counter += 1
                start = time.time()
                feed = {model.inputs: x,
                       model.targets: y,
                       model.init_state: new_state,
                       model.keep_prob: keep_prob}
                batch_loss, new_state, _ = sess.run([model.loss, model.state, model.optimizer],
                                                   feed_dict = feed)
                end = time.time()
                if counter % 500 == 0:
                    print("Training epoch: {}/{}***".format(e+1, epochs),
                         "Training steps: {}***".format(counter),
                         "Training loss: {:.4f}***".format(batch_loss),
                         "{:.4f} sec/per batch".format((end-start)))
                if (counter % n_save == 0):
                    saver.save(sess, "checkpoint/checkpoint_{}.ckpt".format(counter))
        saver.save(sess, "checkpoint/checkpoint_{}.ckpt".format(counter))

def pick_top_k(preds, vocab_size, top_k=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_k]] = 0
    p = p/np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def gentext(checkpoint, n_samples, hidden_units, vocab_to_int, 
            int_to_vocab, vocab_size, prime='the '):
    text = list(prime)
    model = bytemLSTM(vocab_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.init_state)
        for c in list(prime):
            x = np.zeros((1,1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.init_state: new_state,
                    model.keep_prob:1.0}
            preds, new_state = sess.run([model.pred, model.state],
                                        feed_dict=feed)
        c = pick_top_k(preds, vocab_size)
        text.append(int_to_vocab[c])
        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.init_state: new_state,
                    model.keep_prob: 1.0}
            preds, new_state = sess.run([model.pred, model.state],
                                        feed_dict=feed)
            c = pick_top_k(preds, vocab_size)
            text.append(int_to_vocab[c])
    return ''.join(text)


def feature_extract(checkpoint, text, hidden_units, vocab_to_int, 
            int_to_vocab, vocab_size):
    text = list(text)
    model = bytemLSTM(vocab_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.init_state)
        for c in list(text):
            x = np.zeros((1,1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.init_state: new_state,
                    model.keep_prob:1.0}
            new_state = sess.run(model.state, feed_dict=feed)
    return new_state

