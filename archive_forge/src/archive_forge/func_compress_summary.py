import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.preprocessing.tf_data_layer import TFDataLayer
from keras.src.utils import argument_validation
from keras.src.utils import numerical_utils
from keras.src.utils.module_utils import tensorflow as tf
def compress_summary(summary, epsilon):
    """Compress a summary to within `epsilon` accuracy.

    The compression step is needed to keep the summary sizes small after
    merging, and also used to return the final target boundaries. It finds the
    new bins based on interpolating cumulative weight percentages from the large
    summary.  Taking the difference of the cumulative weights from the previous
    bin's cumulative weight will give the new weight for that bin.

    Args:
        summary: 2D `np.ndarray` summary to be compressed.
        epsilon: A `'float32'` that determines the approximate desired
        precision.

    Returns:
        A 2D `np.ndarray` that is a compressed summary. First column is the
        interpolated partition values, the second is the weights (counts).
    """
    if summary.shape[1] * epsilon < 1:
        return summary
    percents = epsilon + np.arange(0.0, 1.0, epsilon)
    cum_weights = summary[1].cumsum()
    cum_weight_percents = cum_weights / cum_weights[-1]
    new_bins = np.interp(percents, cum_weight_percents, summary[0])
    cum_weights = np.interp(percents, cum_weight_percents, cum_weights)
    new_weights = cum_weights - np.concatenate((np.array([0]), cum_weights[:-1]))
    summary = np.stack((new_bins, new_weights))
    return summary.astype('float32')