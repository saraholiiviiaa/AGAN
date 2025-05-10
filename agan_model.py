import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import tensorflow as tf
from tensorflow.keras import layers, Model
from layers.ghost_module import GhostModule
from layers.aca import AdaptiveContextualAttention as ACA
from layers.iaca import InputAwareComplexityAdjustment as IACA
from layers.hal import HybridActivationLayer as HAL
from arcface import ArcMarginProduct

def build_agan_model(input_shape=(112, 112, 3), num_classes=17, embedding_size=256):
    image_input = layers.Input(shape=input_shape, name='image_input')

    x = GhostModule(16)(image_input)
    x = ACA()(x)
    x = IACA()(x)
    x = HAL()(x)
    x = layers.DepthwiseConv2D(kernel_size=7, strides=1, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(embedding_size)(x)
    x = tf.nn.l2_normalize(x, axis=1, name='l2_norm')

    label_input = layers.Input(shape=(), name='label_input')
    output = ArcMarginProduct(n_classes=num_classes)([x, label_input])

    model = Model(inputs=[image_input, label_input], outputs=output, name='agan_arcface_model')
    return model
