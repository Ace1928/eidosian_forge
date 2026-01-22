import contextlib
import threading
from typing import Any, Callable, Optional, Sequence
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.pack', v1=[])
def pack(tensors: Sequence[Any], layout: layout_lib.Layout) -> Any:
    """Packs `tf.Tensor` components into a DTensor.

  Packing and unpacking are inverse operations:

  ```
  * unpack(pack(tensors)) == tensors
  * pack(unpack(dtensor)) == dtensor
  ```

  1. For any DTensor on the mesh, `unpack` returns the raw components placed on
     each underlying device.
  2. Packing these raw components in the same order using `pack` returns a
     DTensor which should be identical to the original DTensor--both the content
     value and the layout.

  **Shape, Rank, and Scalars**: The rank of the DTensor is the same as the
  rank of its raw components, i.e., rank is preserved.  This leads to a
  consistent interpretation for packing scalar values into a DTensor. The only
  valid layout for a scalar value is fully replicated, and the individual
  components must be identical scalars.

  Each input `tensors[i]` will be copied to `layout.mesh.local_device[i]`
  if not already on the local device. Non-local components should not be passed
  to `pack`; use `copy_to_mesh` and `relayout` to place tensors on all global
  devices on a mesh.

  It is the caller's responsibility to ensure that the underlying values
  for `pack` adhere to the specified layout, and that only as many values are
  specified as there are local devices. Pack does not move data between clients.
  See examples below for more detail about layouts.

  For example, assume we have a mesh `[X(2), Y(3)]`, which has in total 6
  underlying devices. Futuremore, assume that the device location mapping is
  the following:

  ```
  device_ID  |  location X, Y
          0     0, 0
          1     0, 1
          2     0, 2
          3     1, 0
          4     1, 1
          5     1, 2
  ```

  1. For 1-D vector DTensor with shape `[128]` with layout `[mesh.X]` and value
     as `range(128)`, the raw components will have shape `[64]` each, and the
     raw components will be:

     ```
     device_ID  |  raw component
             0     range(0, 64)
             1     range(0, 64)
             2     range(0, 64)
             3     range(64, 128)
             4     range(64, 128)
             5     range(64, 128)
     ```

     This also means for a 1-D DTensor with shape `[2]` and layout `[mesh.X]`,
     the raw components have shape `[1]` rather than the shape for scalar values
     `[]`.

  2. For 2-D vector DTensor with shape `[2, 3]` with layout `[mesh.X, mesh.Y]`
     and value as `range(6)`, this is basically a fully-sharded DTensor.

     From global view, the content looks like
     ```
     [
       [0.0, 1.0, 2.0],
       [3.0, 4.0, 5.0],
     ]
     ```

     The raw components will have shape `[1, 1]` each, and have the following
     content:

     ```
     device_ID  |  raw component
             0     [[0.0]]
             1     [[1.0]]
             2     [[2.0]]
             3     [[3.0]]
             4     [[4.0]]
             5     [[5.0]]
     ```

  3. For a scalar value `123.0` DTensor, it can only have one legitimate layout
     `[]` (no dimension, but fully replicated).

     The raw components will have shape `[]` each, and have the following
     content:

     ```
     device_ID  |  raw component
             0     123.0
             1     123.0
             2     123.0
             3     123.0
             4     123.0
             5     123.0
     ```

     Again, caller of `pack` is expected to provide 6 identical value raw
     components with scalar shapes.

  4. For 3-D vector DTensor with shape `[2, 2, 3]` with layout
     `[X, unsharded, unsharded]` and value as `range(12)`,

     From global view, the content looks like:
     ```
     [
       [
         [0.0, 1.0, 2.0],
         [3.0, 4.0, 5.0],
       ],
       [
         [6.0, 7.0, 8.0],
         [9.0, 10., 11.],
       ],
     ]
     ```

     The raw components will have shape `[1, 2, 3]` each, and have the following
     content:

     ```
     device_ID  |  raw component
             0     range(6).reshape([1, 2, 3])
             1     range(6).reshape([1, 2, 3])
             2     range(6).reshape([1, 2, 3])
             3     range(6, 12).reshape([1, 2, 3])
             4     range(6, 12).reshape([1, 2, 3])
             5     range(6, 12).reshape([1, 2, 3])
     ```

  Args:
    tensors: The list of local tensor components to pack into a DTensor.
    layout: The layout of the DTensor to be created.

  Returns:
    A DTensor created from the individual component tensors.

  Raises:
    RuntimeError: When `pack` is not called eagerly.
  """
    return _dtensor_device().pack(tensors, layout)