from typing import Union
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.util import _pywrap_determinism
from tensorflow.python.util import _pywrap_tensor_float_32_execution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.experimental.enable_tensor_float_32_execution')
def enable_tensor_float_32_execution(enabled):
    """Enable or disable the use of TensorFloat-32 on supported hardware.

  [TensorFloat-32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format),
  or TF32 for short, is a math mode for NVIDIA Ampere GPUs and above.
  TensorFloat-32 execution causes certain float32 ops, such as matrix
  multiplications and convolutions, to run much faster on such GPUs but with
  reduced precision. This reduced precision should not impact convergence of
  deep learning models in practice.

  TensorFloat-32 is enabled by default. TensorFloat-32 is only supported on
  NVIDIA GPUs starting with the Ampere generation, so older NVIDIA GPUs will use
  the full float32 precision regardless of whether TensorFloat-32 is enabled or
  not. If you want to use the full float32 precision on all GPUs, you can
  disable TensorFloat-32 execution with this function. For example:

  ```python
  x = tf.fill((1024, 1024), 1.0001)
  y = tf.fill((1024, 1024), 1.)
  # TensorFloat-32 is enabled, so matmul is run with reduced precision
  print(tf.linalg.matmul(x, y)[0, 0])  # 1024.0
  tf.config.experimental.enable_tensor_float_32_execution(False)
  # Matmul is run with full precision
  print(tf.linalg.matmul(x, y)[0, 0])  # ~1024.1
  ```

  To check whether TensorFloat-32 execution is currently enabled, use
  `tf.config.experimental.tensor_float_32_execution_enabled`.

  If TensorFloat-32 is enabled, float32 inputs of supported ops, such as
  `tf.linalg.matmul`, will be rounded from 23 bits of precision to 10 bits of
  precision in most cases. This allows the ops to execute much faster by
  utilizing the GPU's tensor cores. TensorFloat-32 has the same dynamic range as
  float32, meaning it is no more likely to underflow or overflow than float32.
  Ops still use float32 accumulation when TensorFloat-32 is enabled. Enabling or
  disabling TensorFloat-32 only affects Ampere GPUs and above.

  Note TensorFloat-32 is not always used in supported ops, as only inputs of
  certain shapes are supported. Support for more input shapes and more ops may
  be added in the future. As a result, precision of float32 ops may decrease in
  minor versions of TensorFlow.

  TensorFloat-32 is also used for some complex64 ops. Currently, TensorFloat-32
  is used in fewer cases for complex64 as it is for float32.

  Simiarly to GPUs, TPUs also run certain float32 ops, like matrix
  multiplications and convolutions, with lower precision by default. Unlike
  GPUs, TPUs use bfloat16 precision instead of TensorFloat-32 precision for such
  ops. Disabling TensorFloat-32 with this function also causes TPUs to run
  float32 ops with the full float32 precision but with lower performance.

  Args:
    enabled: Bool indicating whether to enable TensorFloat-32 execution.
  """
    _pywrap_tensor_float_32_execution.enable(enabled)