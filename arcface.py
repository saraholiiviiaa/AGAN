# ✅ ArcMarginProduct with label dtype fix
import tensorflow as tf
import math

class ArcMarginProduct(tf.keras.layers.Layer):
    def __init__(self, n_classes, s=30.0, m=0.5, easy_margin=False, **kwargs):
        super(ArcMarginProduct, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

    def build(self, input_shape):
        embedding_dim = input_shape[0][-1]
        self.W = self.add_weight(
            name='W',
            shape=(embedding_dim, self.n_classes),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        embeddings, labels = inputs
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        cos_theta = tf.matmul(embeddings, W)
        theta = tf.acos(tf.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = tf.cos(theta + self.m)

        labels = tf.cast(labels, tf.int32)
        one_hot = tf.one_hot(labels, depth=self.n_classes)
        output = cos_theta * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        return output
