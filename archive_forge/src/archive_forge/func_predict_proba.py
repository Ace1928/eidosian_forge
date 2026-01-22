import copy
import warnings
from tensorflow.python import tf2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import layers as layer_module
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import functional
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.module import module
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
def predict_proba(self, x, batch_size=32, verbose=0):
    """Generates class probability predictions for the input samples.

    The input samples are processed batch by batch.

    Args:
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.

    Returns:
        A Numpy array of probability predictions.
    """
    warnings.warn('`model.predict_proba()` is deprecated and will be removed after 2021-01-01. Please use `model.predict()` instead.')
    preds = self.predict(x, batch_size, verbose)
    if preds.min() < 0.0 or preds.max() > 1.0:
        logging.warning('Network returning invalid probability values. The last layer might not normalize predictions into probabilities (like softmax or sigmoid would).')
    return preds