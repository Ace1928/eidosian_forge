from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_utils
def rng_bit_generator(algorithm, initial_state, shape, dtype):
    """Stateless PRNG bit generator.

  Wraps the XLA RngBitGenerator operator, documented at
    https://www.tensorflow.org/performance/xla/operation_semantics#rngbitgenerator.

  Args:
    algorithm: The PRNG algorithm to use, one of tf.random.Algorithm.{PHILOX,
      THREEFRY, AUTO_SELECT}.
    initial_state: Initial state for the PRNG algorithm. For THREEFRY, it should
      be a u64[2] and for PHILOX a u64[3].
    shape: The output shape of the generated data.
    dtype: The type of the tensor.

  Returns:
    a tuple with a new state and generated data of the given shape.
  """
    alg_int = random_ops_util.convert_alg_to_int(algorithm)
    return gen_xla_ops.xla_rng_bit_generator(alg_int, initial_state, shape, dtype=dtype)