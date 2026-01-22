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
def ctc_loss_v2_eager_fallback(inputs: _atypes.TensorFuzzingAnnotation[_atypes.Float32], labels_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], labels_values: _atypes.TensorFuzzingAnnotation[_atypes.Int32], sequence_length: _atypes.TensorFuzzingAnnotation[_atypes.Int32], preprocess_collapse_repeated: bool, ctc_merge_repeated: bool, ignore_longer_outputs_than_inputs: bool, name, ctx):
    if preprocess_collapse_repeated is None:
        preprocess_collapse_repeated = False
    preprocess_collapse_repeated = _execute.make_bool(preprocess_collapse_repeated, 'preprocess_collapse_repeated')
    if ctc_merge_repeated is None:
        ctc_merge_repeated = True
    ctc_merge_repeated = _execute.make_bool(ctc_merge_repeated, 'ctc_merge_repeated')
    if ignore_longer_outputs_than_inputs is None:
        ignore_longer_outputs_than_inputs = False
    ignore_longer_outputs_than_inputs = _execute.make_bool(ignore_longer_outputs_than_inputs, 'ignore_longer_outputs_than_inputs')
    inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
    labels_indices = _ops.convert_to_tensor(labels_indices, _dtypes.int64)
    labels_values = _ops.convert_to_tensor(labels_values, _dtypes.int32)
    sequence_length = _ops.convert_to_tensor(sequence_length, _dtypes.int32)
    _inputs_flat = [inputs, labels_indices, labels_values, sequence_length]
    _attrs = ('preprocess_collapse_repeated', preprocess_collapse_repeated, 'ctc_merge_repeated', ctc_merge_repeated, 'ignore_longer_outputs_than_inputs', ignore_longer_outputs_than_inputs)
    _result = _execute.execute(b'CTCLossV2', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CTCLossV2', _inputs_flat, _attrs, _result)
    _result = _CTCLossV2Output._make(_result)
    return _result