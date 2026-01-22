import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import index_lookup
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
Computes a vocabulary of interger terms from tokens in a dataset.

        Calling `adapt()` on an `IntegerLookup` layer is an alternative to
        passing in a precomputed vocabulary  on construction via the
        `vocabulary` argument.  An `IntegerLookup` layer should always be either
        adapted over a dataset or supplied with a vocabulary.

        During `adapt()`, the layer will build a vocabulary of all integer
        tokens seen in the dataset, sorted by occurrence count, with ties broken
        by sort order of the tokens (high to low). At the end of `adapt()`, if
        `max_tokens` is set, the vocabulary wil be truncated to `max_tokens`
        size. For example, adapting a layer with `max_tokens=1000` will compute
        the 1000 most frequent tokens occurring in the input dataset. If
        `output_mode='tf-idf'`, `adapt()` will also learn the document
        frequencies of each token in the input dataset.

        In order to make `StringLookup` efficient in any distribution context,
        the vocabulary is kept static with respect to any compiled `tf.Graph`s
        that call the layer. As a consequence, if the layer is adapted a second
        time, any models using the layer should be re-compiled. For more
        information see
        `tf.keras.layers.experimental.preprocessing.PreprocessingLayer.adapt`.

        `adapt()` is meant only as a single machine utility to compute layer
        state.  To analyze a dataset that cannot fit on a single machine, see
        [Tensorflow Transform](
        https://www.tensorflow.org/tfx/transform/get_started) for a
        multi-machine, map-reduce solution.

        Arguments:
          data: The data to train on. It can be passed either as a
              `tf.data.Dataset`, or as a numpy array.
          batch_size: Integer or `None`.
              Number of samples per state update.
              If unspecified, `batch_size` will default to 32.
              Do not specify the `batch_size` if your data is in the
              form of datasets, generators, or `keras.utils.Sequence` instances
              (since they generate batches).
          steps: Integer or `None`.
              Total number of steps (batches of samples)
              When training with input tensors such as
              TensorFlow data tensors, the default `None` is equal to
              the number of samples in your dataset divided by
              the batch size, or 1 if that cannot be determined. If x is a
              `tf.data` dataset, and 'steps' is None, the epoch will run until
              the input dataset is exhausted. When passing an infinitely
              repeating dataset, you must specify the `steps` argument. This
              argument is not supported with array inputs.
        