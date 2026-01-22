import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def initialize_table_from_text_file(table_handle: _atypes.TensorFuzzingAnnotation[_atypes.String], filename: _atypes.TensorFuzzingAnnotation[_atypes.String], key_index: int, value_index: int, vocab_size: int=-1, delimiter: str='\t', offset: int=0, name=None):
    """Initializes a table from a text file.

  It inserts one key-value pair into the table for each line of the file.
  The key and value is extracted from the whole line content, elements from the
  split line based on `delimiter` or the line number (starting from zero).
  Where to extract the key and value from a line is specified by `key_index` and
  `value_index`.

  - A value of -1 means use the line number(starting from zero), expects `int64`.
  - A value of -2 means use the whole line content, expects `string`.
  - A value >= 0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      Handle to a table which will be initialized.
    filename: A `Tensor` of type `string`. Filename of a vocabulary text file.
    key_index: An `int` that is `>= -2`.
      Column index in a line to get the table `key` values from.
    value_index: An `int` that is `>= -2`.
      Column index that represents information of a line to get the table
      `value` values from.
    vocab_size: An optional `int` that is `>= -1`. Defaults to `-1`.
      Number of elements of the file, use -1 if unknown.
    delimiter: An optional `string`. Defaults to `"\\t"`.
      Delimiter to separate fields in a line.
    offset: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("initialize_table_from_text_file op does not support eager execution. Arg 'table_handle' is a ref.")
    key_index = _execute.make_int(key_index, 'key_index')
    value_index = _execute.make_int(value_index, 'value_index')
    if vocab_size is None:
        vocab_size = -1
    vocab_size = _execute.make_int(vocab_size, 'vocab_size')
    if delimiter is None:
        delimiter = '\t'
    delimiter = _execute.make_str(delimiter, 'delimiter')
    if offset is None:
        offset = 0
    offset = _execute.make_int(offset, 'offset')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('InitializeTableFromTextFile', table_handle=table_handle, filename=filename, key_index=key_index, value_index=value_index, vocab_size=vocab_size, delimiter=delimiter, offset=offset, name=name)
    return _op