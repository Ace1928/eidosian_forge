import functools
import numpy as np
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import cond
from tensorflow.python.util import nest
Verifies that possibly-structured symbol has types compatible vith another.

  See _verify_spec_compatible for a more concrete meaning of "compatible".
  Unspec _verify_spec_compatible, which handles singular Tensor-spec objects,
  verify_structures_compatible can process structures recognized by tf.nest.

  Args:
    input_name: A name to use for `input_` in error messages.
    spec_name: A name to use for `spec` in error messages.
    input_: Any, value to verify. May, but doesn't need to, be a structure.
    spec: Any, value that `input_` must be compatible with. May, but doesn't
      need to, be a structure.

  Raises:
    ValueError if the two types have been determined not to be compatible.
  