import os
import warnings
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import model_config
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving import utils_v1 as model_utils
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.util import compat
from tensorflow.python.util import nest
def load_from_saved_model(saved_model_path, custom_objects=None):
    """Loads a keras Model from a SavedModel created by `export_saved_model()`.

  This function reinstantiates model state by:
  1) loading model topology from json (this will eventually come
     from metagraph).
  2) loading model weights from checkpoint.

  Example:

  ```python
  import tensorflow as tf

  # Create a tf.keras model.
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(1, input_shape=[10]))
  model.summary()

  # Save the tf.keras model in the SavedModel format.
  path = '/tmp/simple_keras_model'
  tf.keras.experimental.export_saved_model(model, path)

  # Load the saved keras model back.
  new_model = tf.keras.experimental.load_from_saved_model(path)
  new_model.summary()
  ```

  Args:
    saved_model_path: a string specifying the path to an existing SavedModel.
    custom_objects: Optional dictionary mapping names
        (strings) to custom classes or functions to be
        considered during deserialization.

  Returns:
    a keras.Model instance.
  """
    warnings.warn('`tf.keras.experimental.load_from_saved_model` is deprecatedand will be removed in a future version. Please switch to `tf.keras.models.load_model`.')
    model_json_filepath = os.path.join(compat.as_bytes(saved_model_path), compat.as_bytes(constants.ASSETS_DIRECTORY), compat.as_bytes(SAVED_MODEL_FILENAME_JSON))
    with gfile.Open(model_json_filepath, 'r') as f:
        model_json = f.read()
    model = model_config.model_from_json(model_json, custom_objects=custom_objects)
    checkpoint_prefix = os.path.join(compat.as_text(saved_model_path), compat.as_text(constants.VARIABLES_DIRECTORY), compat.as_text(constants.VARIABLES_FILENAME))
    model.load_weights(checkpoint_prefix)
    return model