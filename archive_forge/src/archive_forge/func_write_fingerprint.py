from absl import logging
from tensorflow.core.config import flags
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as fingerprinting_pywrap
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import compat
def write_fingerprint(export_dir):
    """Write fingerprint protobuf, if requested.

  Writes a `tf.saved_model.experimental.Fingerprint` object to a
  `fingerprint.pb` file in the `export_dir` using the `saved_model.pb` file
  contained in `export_dir`.

  Args:
    export_dir: The directory in which to write the fingerprint.
  """
    if flags.config().saved_model_fingerprinting.value():
        fingerprint_path = file_io.join(compat.as_str(export_dir), compat.as_str(constants.FINGERPRINT_FILENAME))
        logging.info('Writing fingerprint to %s', fingerprint_path)
        try:
            fingerprint_serialized = fingerprinting_pywrap.CreateFingerprintDef(export_dir)
        except fingerprinting_pywrap.FingerprintException as e:
            raise ValueError(e) from None
        file_io.atomic_write_string_to_file(fingerprint_path, fingerprint_serialized)
        metrics.SetWriteFingerprint(fingerprint=fingerprint_serialized)
        metrics.SetWritePathAndSingleprint(path=export_dir, singleprint=singleprint_from_saved_model(export_dir))