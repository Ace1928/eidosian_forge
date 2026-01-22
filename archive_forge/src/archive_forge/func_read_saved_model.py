import os
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
def read_saved_model(saved_model_dir):
    """Reads the saved_model.pb or saved_model.pbtxt file containing `SavedModel`.

  Args:
    saved_model_dir: Directory containing the SavedModel file.

  Returns:
    A `SavedModel` protocol buffer.

  Raises:
    IOError: If the file does not exist, or cannot be successfully parsed.
  """
    path_to_pbtxt = os.path.join(compat.as_bytes(saved_model_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
    path_to_pb = os.path.join(compat.as_bytes(saved_model_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
    if not file_io.file_exists(path_to_pbtxt) and (not file_io.file_exists(path_to_pb)):
        raise IOError('SavedModel file does not exist at: %s' % saved_model_dir)
    saved_model = saved_model_pb2.SavedModel()
    if file_io.file_exists(path_to_pb):
        with file_io.FileIO(path_to_pb, 'rb') as f:
            file_content = f.read()
        try:
            saved_model.ParseFromString(file_content)
            return saved_model
        except message.DecodeError as e:
            raise IOError('Cannot parse proto file %s: %s.' % (path_to_pb, str(e)))
    elif file_io.file_exists(path_to_pbtxt):
        with file_io.FileIO(path_to_pbtxt, 'rb') as f:
            file_content = f.read()
        try:
            text_format.Merge(file_content.decode('utf-8'), saved_model)
            return saved_model
        except text_format.ParseError as e:
            raise IOError('Cannot parse pbtxt file %s: %s.' % (path_to_pbtxt, str(e)))
    else:
        raise IOError('SavedModel file does not exist at: %s/{%s|%s}' % (saved_model_dir, constants.SAVED_MODEL_FILENAME_PBTXT, constants.SAVED_MODEL_FILENAME_PB))