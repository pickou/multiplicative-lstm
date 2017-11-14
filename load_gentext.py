#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from mlstm import preprocess, gentext

def load_gentext(filename, prime):
    """ load frome the latest checkpoint, and generate text

    """
    hidden_units = 512
    n_samples = 400

    vocab, vocab_to_int, int_to_vocab, encoded = preprocess(filename)
    checkpoint = tf.train.latest_checkpoint('./checkpoint')
    text = gentext(checkpoint, n_samples, hidden_units, vocab_to_int,
            int_to_vocab, len(vocab), prime=prime)
    print(text)

if __name__ == "__main__":
    load_gentext('anna.txt', u'This morning')

