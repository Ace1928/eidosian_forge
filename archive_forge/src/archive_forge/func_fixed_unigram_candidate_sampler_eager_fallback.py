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
def fixed_unigram_candidate_sampler_eager_fallback(true_classes: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_true: int, num_sampled: int, unique: bool, range_max: int, vocab_file: str, distortion: float, num_reserved_ids: int, num_shards: int, shard: int, unigrams, seed: int, seed2: int, name, ctx):
    num_true = _execute.make_int(num_true, 'num_true')
    num_sampled = _execute.make_int(num_sampled, 'num_sampled')
    unique = _execute.make_bool(unique, 'unique')
    range_max = _execute.make_int(range_max, 'range_max')
    if vocab_file is None:
        vocab_file = ''
    vocab_file = _execute.make_str(vocab_file, 'vocab_file')
    if distortion is None:
        distortion = 1
    distortion = _execute.make_float(distortion, 'distortion')
    if num_reserved_ids is None:
        num_reserved_ids = 0
    num_reserved_ids = _execute.make_int(num_reserved_ids, 'num_reserved_ids')
    if num_shards is None:
        num_shards = 1
    num_shards = _execute.make_int(num_shards, 'num_shards')
    if shard is None:
        shard = 0
    shard = _execute.make_int(shard, 'shard')
    if unigrams is None:
        unigrams = []
    if not isinstance(unigrams, (list, tuple)):
        raise TypeError("Expected list for 'unigrams' argument to 'fixed_unigram_candidate_sampler' Op, not %r." % unigrams)
    unigrams = [_execute.make_float(_f, 'unigrams') for _f in unigrams]
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    true_classes = _ops.convert_to_tensor(true_classes, _dtypes.int64)
    _inputs_flat = [true_classes]
    _attrs = ('num_true', num_true, 'num_sampled', num_sampled, 'unique', unique, 'range_max', range_max, 'vocab_file', vocab_file, 'distortion', distortion, 'num_reserved_ids', num_reserved_ids, 'num_shards', num_shards, 'shard', shard, 'unigrams', unigrams, 'seed', seed, 'seed2', seed2)
    _result = _execute.execute(b'FixedUnigramCandidateSampler', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FixedUnigramCandidateSampler', _inputs_flat, _attrs, _result)
    _result = _FixedUnigramCandidateSamplerOutput._make(_result)
    return _result