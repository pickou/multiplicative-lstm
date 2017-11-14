#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from mlstm import preprocess, Train, gentext

def train_gentext(filename, prime):
    """ train the mlstm and generate new text with prime

    """
    batch_size = 64 
    num_steps = 100
    hidden_units = 512
    lr = 0.005
    keep_prob = 0.5
    epochs = 50 
    n_save = 1500
    n_samples = 400

    vocab, vocab_to_int, int_to_vocab, encoded = preprocess(filename)
    Train(vocab, encoded, epochs, n_save, batch_size, num_steps, keep_prob)
    checkpoint = tf.train.latest_checkpoint('./checkpoint')
    text = gentext(checkpoint, n_samples, hidden_units, vocab_to_int,
            int_to_vocab, len(vocab), prime=prime)
    print(text)

if __name__ == "__main__":
    train_gentext('anna.txt', 'This morning')
