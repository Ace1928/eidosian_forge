from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl as loader
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
Saves the updated `SavedModel`.

    Args:
      new_export_dir: Path where the updated `SavedModel` will be saved. If
          None, the input `SavedModel` will be overriden with the updates.

    Raises:
      errors.OpError: If there are errors during the file save operation.
    