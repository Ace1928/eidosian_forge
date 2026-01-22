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
def generate_vocab_remapping_eager_fallback(new_vocab_file: _atypes.TensorFuzzingAnnotation[_atypes.String], old_vocab_file: _atypes.TensorFuzzingAnnotation[_atypes.String], new_vocab_offset: int, num_new_vocab: int, old_vocab_size: int, name, ctx):
    new_vocab_offset = _execute.make_int(new_vocab_offset, 'new_vocab_offset')
    num_new_vocab = _execute.make_int(num_new_vocab, 'num_new_vocab')
    if old_vocab_size is None:
        old_vocab_size = -1
    old_vocab_size = _execute.make_int(old_vocab_size, 'old_vocab_size')
    new_vocab_file = _ops.convert_to_tensor(new_vocab_file, _dtypes.string)
    old_vocab_file = _ops.convert_to_tensor(old_vocab_file, _dtypes.string)
    _inputs_flat = [new_vocab_file, old_vocab_file]
    _attrs = ('new_vocab_offset', new_vocab_offset, 'num_new_vocab', num_new_vocab, 'old_vocab_size', old_vocab_size)
    _result = _execute.execute(b'GenerateVocabRemapping', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('GenerateVocabRemapping', _inputs_flat, _attrs, _result)
    _result = _GenerateVocabRemappingOutput._make(_result)
    return _result