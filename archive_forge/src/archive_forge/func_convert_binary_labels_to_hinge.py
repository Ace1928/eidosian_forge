import warnings
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.losses.loss import Loss
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.saving import serialization_lib
from keras.src.utils.numerical_utils import normalize
def convert_binary_labels_to_hinge(y_true):
    """Converts binary labels into -1/1 for hinge loss/metric calculation."""
    are_zeros = ops.equal(y_true, 0)
    are_ones = ops.equal(y_true, 1)
    is_binary = ops.all(ops.logical_or(are_zeros, are_ones))

    def _convert_binary_labels():
        return 2.0 * y_true - 1.0

    def _return_labels_unconverted():
        return y_true
    updated_y_true = ops.cond(is_binary, _convert_binary_labels, _return_labels_unconverted)
    return updated_y_true