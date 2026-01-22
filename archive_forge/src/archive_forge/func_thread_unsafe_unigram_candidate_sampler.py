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
def thread_unsafe_unigram_candidate_sampler(true_classes: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, seed: int=0, seed2: int=0, name=None):
    """Generates labels for candidate sampling with a learned unigram distribution.

  See explanations of candidate sampling and the data formats at
  go/candidate-sampling.

  For each batch, this op picks a single set of sampled candidate labels.

  The advantages of sampling candidates per-batch are simplicity and the
  possibility of efficient dense matrix multiplication. The disadvantage is that
  the sampled candidates must be chosen independently of the context and of the
  true labels.

  Args:
    true_classes: A `Tensor` of type `int64`.
      A batch_size * num_true matrix, in which each row contains the
      IDs of the num_true target_classes in the corresponding original label.
    num_true: An `int` that is `>= 1`. Number of true labels per context.
    num_sampled: An `int` that is `>= 1`.
      Number of candidates to randomly sample.
    unique: A `bool`.
      If unique is true, we sample with rejection, so that all sampled
      candidates in a batch are unique. This requires some approximation to
      estimate the post-rejection sampling probabilities.
    range_max: An `int` that is `>= 1`.
      The sampler will sample integers from the interval [0, range_max).
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sampled_candidates, true_expected_count, sampled_expected_count).

    sampled_candidates: A `Tensor` of type `int64`.
    true_expected_count: A `Tensor` of type `float32`.
    sampled_expected_count: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ThreadUnsafeUnigramCandidateSampler', name, true_classes, 'num_true', num_true, 'num_sampled', num_sampled, 'unique', unique, 'range_max', range_max, 'seed', seed, 'seed2', seed2)
            _result = _ThreadUnsafeUnigramCandidateSamplerOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return thread_unsafe_unigram_candidate_sampler_eager_fallback(true_classes, num_true=num_true, num_sampled=num_sampled, unique=unique, range_max=range_max, seed=seed, seed2=seed2, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_true = _execute.make_int(num_true, 'num_true')
    num_sampled = _execute.make_int(num_sampled, 'num_sampled')
    unique = _execute.make_bool(unique, 'unique')
    range_max = _execute.make_int(range_max, 'range_max')
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ThreadUnsafeUnigramCandidateSampler', true_classes=true_classes, num_true=num_true, num_sampled=num_sampled, unique=unique, range_max=range_max, seed=seed, seed2=seed2, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_true', _op._get_attr_int('num_true'), 'num_sampled', _op._get_attr_int('num_sampled'), 'unique', _op._get_attr_bool('unique'), 'range_max', _op._get_attr_int('range_max'), 'seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ThreadUnsafeUnigramCandidateSampler', _inputs_flat, _attrs, _result)
    _result = _ThreadUnsafeUnigramCandidateSamplerOutput._make(_result)
    return _result