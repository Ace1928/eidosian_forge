import random
import tempfile
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import string_lookup
def create_serving_signature(self, model, feature_mapper, label_inverse_lookup_layer):
    """Create serving signature for the given model."""

    @tf.function
    def serve_fn(raw_features):
        raw_features = tf.expand_dims(raw_features, axis=0)
        transformed_features = model.feature_mapper(raw_features)
        outputs = model(transformed_features)
        outputs = tf.squeeze(outputs, axis=0)
        outputs = tf.cast(tf.greater(outputs, 0.5), tf.int64)
        decoded_outputs = model.label_inverse_lookup_layer(outputs)
        return tf.squeeze(decoded_outputs, axis=0)
    model.feature_mapper = feature_mapper
    model.label_inverse_lookup_layer = label_inverse_lookup_layer
    return serve_fn.get_concrete_function(tf.TensorSpec(shape=3, dtype=tf.string, name='example'))