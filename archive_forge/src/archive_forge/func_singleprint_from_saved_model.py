from absl import logging
from tensorflow.core.config import flags
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as fingerprinting_pywrap
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import compat
def singleprint_from_saved_model(export_dir):
    """Returns the singleprint of a SavedModel in `export_dir`.

  Args:
    export_dir: The directory that contains the SavedModel.

  Returns:
    A string containing the singleprint of the SavedModel in `export_dir`.

  Raises:
    ValueError: If a valid singleprint cannot be constructed from a SavedModel.
  """
    try:
        return fingerprinting_pywrap.SingleprintFromSM(export_dir)
    except fingerprinting_pywrap.FingerprintException as e:
        raise ValueError(e) from None