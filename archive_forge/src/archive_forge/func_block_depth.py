import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export
@property
def block_depth(self):
    """Depth of recursively defined circulant blocks defining this `Operator`.

    With `A` the dense representation of this `Operator`,

    `block_depth = 1` means `A` is symmetric circulant.  For example,

    ```
    A = |w z y x|
        |x w z y|
        |y x w z|
        |z y x w|
    ```

    `block_depth = 2` means `A` is block symmetric circulant with symmetric
    circulant blocks.  For example, with `W`, `X`, `Y`, `Z` symmetric circulant,

    ```
    A = |W Z Y X|
        |X W Z Y|
        |Y X W Z|
        |Z Y X W|
    ```

    `block_depth = 3` means `A` is block symmetric circulant with block
    symmetric circulant blocks.

    Returns:
      Python `integer`.
    """
    return self._block_depth