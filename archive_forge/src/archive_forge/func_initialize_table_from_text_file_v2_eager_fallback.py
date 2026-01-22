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
def initialize_table_from_text_file_v2_eager_fallback(table_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], filename: _atypes.TensorFuzzingAnnotation[_atypes.String], key_index: int, value_index: int, vocab_size: int, delimiter: str, offset: int, name, ctx):
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
    table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
    filename = _ops.convert_to_tensor(filename, _dtypes.string)
    _inputs_flat = [table_handle, filename]
    _attrs = ('key_index', key_index, 'value_index', value_index, 'vocab_size', vocab_size, 'delimiter', delimiter, 'offset', offset)
    _result = _execute.execute(b'InitializeTableFromTextFileV2', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result