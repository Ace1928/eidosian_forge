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
def generate_vocab_remapping(new_vocab_file: _atypes.TensorFuzzingAnnotation[_atypes.String], old_vocab_file: _atypes.TensorFuzzingAnnotation[_atypes.String], new_vocab_offset: int, num_new_vocab: int, old_vocab_size: int=-1, name=None):
    """Given a path to new and old vocabulary files, returns a remapping Tensor of

  length `num_new_vocab`, where `remapping[i]` contains the row number in the old
  vocabulary that corresponds to row `i` in the new vocabulary (starting at line
  `new_vocab_offset` and up to `num_new_vocab` entities), or `-1` if entry `i`
  in the new vocabulary is not in the old vocabulary.  The old vocabulary is
  constrained to the first `old_vocab_size` entries if `old_vocab_size` is not the
  default value of -1.

  `num_vocab_offset` enables
  use in the partitioned variable case, and should generally be set through
  examining partitioning info.  The format of the files should be a text file,
  with each line containing a single entity within the vocabulary.

  For example, with `new_vocab_file` a text file containing each of the following
  elements on a single line: `[f0, f1, f2, f3]`, old_vocab_file = [f1, f0, f3],
  `num_new_vocab = 3, new_vocab_offset = 1`, the returned remapping would be
  `[0, -1, 2]`.

  The op also returns a count of how many entries in the new vocabulary
  were present in the old vocabulary, which is used to calculate the number of
  values to initialize in a weight matrix remapping

  This functionality can be used to remap both row vocabularies (typically,
  features) and column vocabularies (typically, classes) from TensorFlow
  checkpoints.  Note that the partitioning logic relies on contiguous vocabularies
  corresponding to div-partitioned variables.  Moreover, the underlying remapping
  uses an IndexTable (as opposed to an inexact CuckooTable), so client code should
  use the corresponding index_table_from_file() as the FeatureColumn framework
  does (as opposed to tf.feature_to_id(), which uses a CuckooTable).

  Args:
    new_vocab_file: A `Tensor` of type `string`. Path to the new vocab file.
    old_vocab_file: A `Tensor` of type `string`. Path to the old vocab file.
    new_vocab_offset: An `int` that is `>= 0`.
      How many entries into the new vocab file to start reading.
    num_new_vocab: An `int` that is `>= 0`.
      Number of entries in the new vocab file to remap.
    old_vocab_size: An optional `int` that is `>= -1`. Defaults to `-1`.
      Number of entries in the old vocab file to consider.  If -1,
      use the entire old vocabulary.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (remapping, num_present).

    remapping: A `Tensor` of type `int64`.
    num_present: A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'GenerateVocabRemapping', name, new_vocab_file, old_vocab_file, 'new_vocab_offset', new_vocab_offset, 'num_new_vocab', num_new_vocab, 'old_vocab_size', old_vocab_size)
            _result = _GenerateVocabRemappingOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return generate_vocab_remapping_eager_fallback(new_vocab_file, old_vocab_file, new_vocab_offset=new_vocab_offset, num_new_vocab=num_new_vocab, old_vocab_size=old_vocab_size, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    new_vocab_offset = _execute.make_int(new_vocab_offset, 'new_vocab_offset')
    num_new_vocab = _execute.make_int(num_new_vocab, 'num_new_vocab')
    if old_vocab_size is None:
        old_vocab_size = -1
    old_vocab_size = _execute.make_int(old_vocab_size, 'old_vocab_size')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('GenerateVocabRemapping', new_vocab_file=new_vocab_file, old_vocab_file=old_vocab_file, new_vocab_offset=new_vocab_offset, num_new_vocab=num_new_vocab, old_vocab_size=old_vocab_size, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('new_vocab_offset', _op._get_attr_int('new_vocab_offset'), 'num_new_vocab', _op._get_attr_int('num_new_vocab'), 'old_vocab_size', _op._get_attr_int('old_vocab_size'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('GenerateVocabRemapping', _inputs_flat, _attrs, _result)
    _result = _GenerateVocabRemappingOutput._make(_result)
    return _result