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
def string_n_grams_eager_fallback(data: _atypes.TensorFuzzingAnnotation[_atypes.String], data_splits: _atypes.TensorFuzzingAnnotation[TV_StringNGrams_Tsplits], separator: str, ngram_widths, left_pad: str, right_pad: str, pad_width: int, preserve_short_sequences: bool, name, ctx):
    separator = _execute.make_str(separator, 'separator')
    if not isinstance(ngram_widths, (list, tuple)):
        raise TypeError("Expected list for 'ngram_widths' argument to 'string_n_grams' Op, not %r." % ngram_widths)
    ngram_widths = [_execute.make_int(_i, 'ngram_widths') for _i in ngram_widths]
    left_pad = _execute.make_str(left_pad, 'left_pad')
    right_pad = _execute.make_str(right_pad, 'right_pad')
    pad_width = _execute.make_int(pad_width, 'pad_width')
    preserve_short_sequences = _execute.make_bool(preserve_short_sequences, 'preserve_short_sequences')
    _attr_Tsplits, (data_splits,) = _execute.args_to_matching_eager([data_splits], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int64)
    data = _ops.convert_to_tensor(data, _dtypes.string)
    _inputs_flat = [data, data_splits]
    _attrs = ('separator', separator, 'ngram_widths', ngram_widths, 'left_pad', left_pad, 'right_pad', right_pad, 'pad_width', pad_width, 'preserve_short_sequences', preserve_short_sequences, 'Tsplits', _attr_Tsplits)
    _result = _execute.execute(b'StringNGrams', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StringNGrams', _inputs_flat, _attrs, _result)
    _result = _StringNGramsOutput._make(_result)
    return _result