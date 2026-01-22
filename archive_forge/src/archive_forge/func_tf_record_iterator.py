from tensorflow.python.lib.io import _pywrap_record_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['io.tf_record_iterator', 'python_io.tf_record_iterator'])
@deprecation.deprecated(date=None, instructions='Use eager execution and: \n`tf.data.TFRecordDataset(path)`')
def tf_record_iterator(path, options=None):
    """An iterator that read the records from a TFRecords file.

  Args:
    path: The path to the TFRecords file.
    options: (optional) A TFRecordOptions object.

  Returns:
    An iterator of serialized TFRecords.

  Raises:
    IOError: If `path` cannot be opened for reading.
  """
    compression_type = TFRecordOptions.get_compression_type_string(options)
    return _pywrap_record_io.RecordIterator(path, compression_type)