#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from mlstm import preprocess, gentext

def load_gentext(filename, prime):
    """ load frome the latest checkpoint, and generate text

    """
    batch_size = 64
    num_steps = 100
    hidden_units = 512
    lr = 5e-4
    keep_prob = 0.5
    epochs = 2
    n_save = 1500
    n_samples = 400

    vocab, vocab_to_int, int_to_vocab, encoded = preprocess(filename)
    checkpoint = tf.train.latest_checkpoint('./checkpoint')
    text = gentext(checkpoint, n_samples, hidden_units, vocab_to_int,
            int_to_vocab, len(vocab), prime=prime)
    print text

if __name__ == "__main__":
    load_gentext('anna.txt', u'This morning')

