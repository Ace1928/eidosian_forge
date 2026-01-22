from tensorflow.python.lib.io import _pywrap_record_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@classmethod
def get_compression_type_string(cls, options):
    """Convert various option types to a unified string.

    Args:
      options: `TFRecordOption`, `TFRecordCompressionType`, or string.

    Returns:
      Compression type as string (e.g. `'ZLIB'`, `'GZIP'`, or `''`).

    Raises:
      ValueError: If compression_type is invalid.
    """
    if not options:
        return ''
    elif isinstance(options, TFRecordOptions):
        return cls.get_compression_type_string(options.compression_type)
    elif isinstance(options, TFRecordCompressionType):
        return cls.compression_type_map[options]
    elif options in TFRecordOptions.compression_type_map:
        return cls.compression_type_map[options]
    elif options in TFRecordOptions.compression_type_map.values():
        return options
    else:
        raise ValueError('Not a valid compression_type: "{}"'.format(options))