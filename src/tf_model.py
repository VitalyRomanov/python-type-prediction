from copy import copy

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.linalg import toeplitz
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D


# alternative models
# https://github.com/flairNLP/flair/tree/master/flair/models
# https://github.com/dhiraa/tener/tree/master/src/tener/models
# https://arxiv.org/pdf/1903.07785v1.pdf
# https://github.com/tensorflow/models/tree/master/research/cvt_text/model


class DefaultEmbedding(Model):
    """
    Creates an embedder that provides the default value for the index -1. The default value is a zero-vector
    """
    def __init__(self, init_vectors=None, shape=None, trainable=True):
        super(DefaultEmbedding, self).__init__()

        if init_vectors is not None:
            self.embs = tf.Variable(init_vectors, dtype=tf.float32,
                           trainable=trainable, name="default_embedder_var")
            shape = init_vectors.shape
        else:
            self.embs = tf.Variable(tf.random.uniform(shape=(shape[0], shape[1]), dtype=tf.float32),
                               name="default_embedder_pad")
        self.pad = tf.Variable(tf.random.uniform(shape=(1, shape[1]), dtype=tf.float32),
                               name="default_embedder_pad")


    def __call__(self, ids):
        emb_matr = tf.concat([self.embs, self.pad], axis=0)
        return tf.nn.embedding_lookup(params=emb_matr, ids=ids)


class PositionalEncoding(Model):
    def __init__(self, seq_len, pos_emb_size):
        super(PositionalEncoding, self).__init__()

        positions = list(range(seq_len * 2))
        position_splt = positions[:seq_len]
        position_splt.reverse()
        self.position_encoding = tf.constant(toeplitz(position_splt, positions[seq_len:]),
                                        dtype=tf.int32,
                                        name="position_encoding")
        self.position_embedding = tf.Variable(tf.random.uniform(shape=(seq_len * 2, pos_emb_size), dtype=tf.float32),
                               name="position_embedding")

    def __call__(self):
        return tf.nn.embedding_lookup(self.position_embedding, self.position_encoding, name="position_lookup")


class TextCnnLayer(Model):
    def __init__(self, out_dim, kernel_shape, activation=None):
        super(TextCnnLayer, self).__init__()

        self.kernel_shape = kernel_shape
        self.out_dim = out_dim

        self.textConv = Conv2D(filters=out_dim, kernel_size=kernel_shape,
                                  activation=activation, data_format='channels_last')

        padding_size = (self.kernel_shape[0] - 1) // 2
        assert padding_size * 2 + 1 == self.kernel_shape[0]
        self.pad_constant = tf.constant([[0, 0], [padding_size, padding_size], [0, 0], [0, 0]])

    def __call__(self, x):
        padded = tf.pad(x, self.pad_constant)
        convolve = self.textConv(padded)
        return tf.squeeze(convolve, axis=-2)


class TextCnn(Model):
    def __init__(self, input_size, h_sizes, seq_len,
                 pos_emb_size, cnn_win_size, dense_size, num_classes,
                 activation=None, dense_activation=None, drop_rate=0.2):
        super(TextCnn, self).__init__()

        self.seq_len = seq_len
        self.h_sizes = h_sizes
        self.dense_size = dense_size
        self.num_classes = num_classes

        kernel_sizes = copy(h_sizes)
        kernel_sizes.pop(-1)
        kernel_sizes.insert(0, input_size)
        kernel_sizes = [(cnn_win_size, ks) for ks in kernel_sizes]

        self.layers_tok = [ TextCnnLayer(out_dim=h_size, kernel_shape=kernel_size, activation=activation)
            for h_size, kernel_size in zip(h_sizes, kernel_sizes)]

        self.layers_pos = [TextCnnLayer(out_dim=h_size, kernel_shape=(cnn_win_size, pos_emb_size), activation=activation)
                       for h_size, _ in zip(h_sizes, kernel_sizes)]

        if dense_activation is None:
            dense_activation = activation

        self.dense_1 = Dense(dense_size, activation=dense_activation)
        self.dropout_1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dense_2 = Dense(num_classes, activation=None) # logits
        self.dropout_2 = tf.keras.layers.Dropout(rate=drop_rate)

    def __call__(self, embs, training=True):

        temp_cnn_emb = embs

        for l in self.layers_tok:
            temp_cnn_emb = l(tf.expand_dims(temp_cnn_emb, axis=3))

        cnn_pool_features = temp_cnn_emb

        token_features = tf.reshape(cnn_pool_features, shape=(-1, self.h_sizes[-1]))

        local_h2 = self.dense_1(token_features)
        tag_logits = self.dense_2(local_h2)

        return tf.reshape(tag_logits, (-1, self.seq_len, self.num_classes))


class TypePredictor(Model):
    def __init__(self, tok_embedder, graph_embedder, train_embeddings=False,
                 h_sizes=[500], dense_size=100, num_classes=None,
                 seq_len=100, pos_emb_size=30, cnn_win_size=3,
                 crf_transitions=None, suffix_prefix_dims=50, suffix_prefix_buckets=1000):
        super(TypePredictor, self).__init__()
        assert num_classes is not None, "set num_classes"

        self.seq_len = seq_len
        self.transition_params = crf_transitions

        with tf.device('/CPU:0'):
            self.tok_emb = DefaultEmbedding(init_vectors=tok_embedder.e, trainable=train_embeddings)
            self.graph_emb = DefaultEmbedding(init_vectors=graph_embedder.e, trainable=train_embeddings)
        self.prefix_emb = DefaultEmbedding(shape=(suffix_prefix_buckets, suffix_prefix_dims))
        self.suffix_emb = DefaultEmbedding(shape=(suffix_prefix_buckets, suffix_prefix_dims))

        input_dim = tok_embedder.e.shape[1] + graph_embedder.e.shape[1] + suffix_prefix_dims * 2


        self.text_cnn = TextCnn(input_size=input_dim, h_sizes=h_sizes,
                                seq_len=seq_len, pos_emb_size=pos_emb_size,
                                cnn_win_size=cnn_win_size, dense_size=dense_size,
                                num_classes=num_classes, activation=tf.nn.relu,
                                dense_activation=tf.nn.tanh)

        self.crf_transition_params = None

    def __call__(self, token_ids, prefix_ids, suffix_ids, graph_ids, training=True):

        tok_emb = self.tok_emb(token_ids)
        graph_emb = self.graph_emb(graph_ids)
        prefix_emb = self.prefix_emb(prefix_ids)
        suffix_emb = self.suffix_emb(suffix_ids)

        embs = tf.concat([tok_emb,
                          graph_emb,
                          prefix_emb,
                          suffix_emb], axis=-1)

        logits = self.text_cnn(embs, training=training)

        return logits

    def loss(self, logits, labels, lengths, class_weights=None):
        losses = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, depth=logits.shape[-1]), logits, axis=-1)
        if class_weights is None:
            loss = tf.reduce_mean(tf.boolean_mask(losses, tf.sequence_mask(lengths, self.seq_len)))
        else:
            loss = tf.reduce_mean(tf.boolean_mask(losses * class_weights, tf.sequence_mask(lengths, self.seq_len)))

        return loss

    def score(self, logits, labels, lengths, scorer=None):
        mask = tf.sequence_mask(lengths, self.seq_len)
        true_labels = tf.boolean_mask(labels, mask)
        argmax = tf.math.argmax(logits, axis=-1)
        estimated_labels = tf.cast(tf.boolean_mask(argmax, mask), tf.int32)

        p, r, f1 = scorer(estimated_labels.numpy(), true_labels.numpy())

        return p, r, f1


def estimate_crf_transitions(batches, n_tags):
    transitions = []
    for _, _, labels, lengths in batches:
        _, transition_params = tfa.text.crf_log_likelihood(tf.ones(shape=(labels.shape[0], labels.shape[1], n_tags)), labels, lengths)
        transitions.append(transition_params.numpy())

    return np.stack(transitions, axis=0).mean(axis=0)

# @tf.function
def train_step(model, optimizer, token_ids, prefix, suffix, graph_ids, labels, lengths, class_weights=None, scorer=None):
    with tf.GradientTape() as tape:
        logits = model(token_ids, prefix, suffix, graph_ids, training=True)
        loss = model.loss(logits, labels, lengths, class_weights=class_weights)
        p, r, f1 = model.score(logits, labels, lengths, scorer=scorer)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, p, r, f1

# @tf.function
def test_step(model, token_ids, prefix, suffix, graph_ids, labels, lengths, class_weights=None, scorer=None):
    logits = model(token_ids, prefix, suffix, graph_ids, training=False)
    loss = model.loss(logits, labels, lengths, class_weights=class_weights)
    p, r, f1 = model.score(logits, labels, lengths, scorer=scorer)

    return loss, p, r, f1


def train(model, train_batches, test_batches, epochs, report_every=10, scorer=None, learning_rate=0.01, learning_rate_decay=1.):

    lr = tf.Variable(learning_rate, trainable=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for e in range(epochs):
        losses = []
        ps = []
        rs = []
        f1s = []

        for ind, batch in enumerate(train_batches):
            loss, p, r, f1 = train_step(model=model, optimizer=optimizer, token_ids=batch['tok_ids'],
                                        prefix=batch['prefix'], suffix=batch['suffix'],
                                        graph_ids=batch['graph_ids'],
                                        labels=batch['tags'],
                                        lengths=batch['lens'],
                                        class_weights=batch['class_weights'],
                                        scorer=scorer)
            losses.append(loss.numpy())
            ps.append(p)
            rs.append(r)
            f1s.append(f1)

        for ind, batch in enumerate(test_batches):
            test_loss, test_p, test_r, test_f1 = test_step(model=model, token_ids=batch['tok_ids'],
                                        prefix=batch['prefix'], suffix=batch['suffix'],
                                        graph_ids=batch['graph_ids'],
                                        labels=batch['tags'],
                                        lengths=batch['lens'],
                                        class_weights=batch['class_weights'],
                                        scorer=scorer)

        print(f"Epoch: {e}, Train Loss: {sum(losses) / len(losses)}, Train P: {sum(ps) / len(ps)}, Train R: {sum(rs) / len(rs)}, Train F1: {sum(f1s) / len(f1s)}, "
              f"Test loss: {test_loss}, Test P: {test_p}, Test R: {test_r}, Test F1: {test_f1}")

        lr.assign(lr * learning_rate_decay)