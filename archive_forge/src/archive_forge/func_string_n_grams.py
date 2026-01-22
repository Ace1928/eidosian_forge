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
def string_n_grams(data: _atypes.TensorFuzzingAnnotation[_atypes.String], data_splits: _atypes.TensorFuzzingAnnotation[TV_StringNGrams_Tsplits], separator: str, ngram_widths, left_pad: str, right_pad: str, pad_width: int, preserve_short_sequences: bool, name=None):
    """Creates ngrams from ragged string data.

  This op accepts a ragged tensor with 1 ragged dimension containing only
  strings and outputs a ragged tensor with 1 ragged dimension containing ngrams
  of that string, joined along the innermost axis.

  Args:
    data: A `Tensor` of type `string`.
      The values tensor of the ragged string tensor to make ngrams out of. Must be a
      1D string tensor.
    data_splits: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The splits tensor of the ragged string tensor to make ngrams out of.
    separator: A `string`.
      The string to append between elements of the token. Use "" for no separator.
    ngram_widths: A list of `ints`. The sizes of the ngrams to create.
    left_pad: A `string`.
      The string to use to pad the left side of the ngram sequence. Only used if
      pad_width != 0.
    right_pad: A `string`.
      The string to use to pad the right side of the ngram sequence. Only used if
      pad_width != 0.
    pad_width: An `int`.
      The number of padding elements to add to each side of each
      sequence. Note that padding will never be greater than 'ngram_widths'-1
      regardless of this value. If `pad_width=-1`, then add `max(ngram_widths)-1`
      elements.
    preserve_short_sequences: A `bool`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (ngrams, ngrams_splits).

    ngrams: A `Tensor` of type `string`.
    ngrams_splits: A `Tensor`. Has the same type as `data_splits`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StringNGrams', name, data, data_splits, 'separator', separator, 'ngram_widths', ngram_widths, 'left_pad', left_pad, 'right_pad', right_pad, 'pad_width', pad_width, 'preserve_short_sequences', preserve_short_sequences)
            _result = _StringNGramsOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return string_n_grams_eager_fallback(data, data_splits, separator=separator, ngram_widths=ngram_widths, left_pad=left_pad, right_pad=right_pad, pad_width=pad_width, preserve_short_sequences=preserve_short_sequences, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    separator = _execute.make_str(separator, 'separator')
    if not isinstance(ngram_widths, (list, tuple)):
        raise TypeError("Expected list for 'ngram_widths' argument to 'string_n_grams' Op, not %r." % ngram_widths)
    ngram_widths = [_execute.make_int(_i, 'ngram_widths') for _i in ngram_widths]
    left_pad = _execute.make_str(left_pad, 'left_pad')
    right_pad = _execute.make_str(right_pad, 'right_pad')
    pad_width = _execute.make_int(pad_width, 'pad_width')
    preserve_short_sequences = _execute.make_bool(preserve_short_sequences, 'preserve_short_sequences')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StringNGrams', data=data, data_splits=data_splits, separator=separator, ngram_widths=ngram_widths, left_pad=left_pad, right_pad=right_pad, pad_width=pad_width, preserve_short_sequences=preserve_short_sequences, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('separator', _op.get_attr('separator'), 'ngram_widths', _op.get_attr('ngram_widths'), 'left_pad', _op.get_attr('left_pad'), 'right_pad', _op.get_attr('right_pad'), 'pad_width', _op._get_attr_int('pad_width'), 'preserve_short_sequences', _op._get_attr_bool('preserve_short_sequences'), 'Tsplits', _op._get_attr_type('Tsplits'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StringNGrams', _inputs_flat, _attrs, _result)
    _result = _StringNGramsOutput._make(_result)
    return _result