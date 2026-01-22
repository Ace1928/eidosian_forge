import collections
import os
import time
from tensorflow.python.keras.saving.utils_v1 import export_output as export_output_lib
from tensorflow.python.keras.saving.utils_v1 import mode_keys
from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.keras.saving.utils_v1.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
def get_export_outputs(export_outputs, predictions):
    """Validate export_outputs or create default export_outputs.

  Args:
    export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. Should be a dict or None.
    predictions:  Predictions `Tensor` or dict of `Tensor`.

  Returns:
    Valid export_outputs dict

  Raises:
    TypeError: if export_outputs is not a dict or its values are not
      ExportOutput instances.
  """
    if export_outputs is None:
        default_output = export_output_lib.PredictOutput(predictions)
        export_outputs = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: default_output}
    if not isinstance(export_outputs, dict):
        raise TypeError('export_outputs must be dict, given: {}'.format(export_outputs))
    for v in export_outputs.values():
        if not isinstance(v, export_output_lib.ExportOutput):
            raise TypeError('Values in export_outputs must be ExportOutput objects. Given: {}'.format(export_outputs))
    _maybe_add_default_serving_output(export_outputs)
    return export_outputs