import functools
import threading
from tensorflow.python import tf2
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as tracking
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
def check_graph_consistency(tensor=None, method='add_loss', force_raise=False):
    """Checks that tensors passed to `add_*` method match the Keras graph.

  When one of the `add_*` method is called inside a V2 conditional branch,
  the underlying tensor gets created in a FuncGraph managed by control_flow_v2.
  We need to raise clear error messages in such cases.

  Args:
    tensor: Tensor to check, or `False` if it is known that an error
      should be raised.
    method: Caller method, one of {'add_metric', 'add_loss', 'add_update'}.
    force_raise: If an error should be raised regardless of `tensor`.

  Raises:
    RuntimeError: In case of an out-of-graph tensor.
  """
    if force_raise or (ops.executing_eagerly_outside_functions() and hasattr(tensor, 'graph') and tensor.graph.is_control_flow_graph):
        if method == 'activity_regularizer':
            bad_example = "\n      class TestModel(tf.keras.Model):\n\n        def __init__(self):\n          super(TestModel, self).__init__(name='test_model')\n          self.dense = tf.keras.layers.Dense(2, activity_regularizer='l2')\n\n        def call(self, x, training=None):\n          if training:\n            return self.dense(x)\n          else:\n            return self.dense(x)\n      "
            correct_example = "\n      class TestModel(tf.keras.Model):\n\n        def __init__(self):\n          super(TestModel, self).__init__(name='test_model')\n          self.dense = tf.keras.layers.Dense(2, activity_regularizer='l2')\n\n        def call(self, x, training=None):\n          return self.dense(x)\n      "
            raise RuntimeError('You are using a layer with `activity_regularizer` in a control flow branch, e.g.:\n{bad_example}\nThis is currently not supported. Please move your call to the layer with `activity_regularizer` out of the control flow branch, e.g.:\n{correct_example}\nYou can also resolve this by marking your outer model/layer dynamic (eager-only) by passing `dynamic=True` to the layer constructor. Any kind of control flow is supported with dynamic layers. Note that using `dynamic=True` requires you to implement static shape inference in the `compute_output_shape(input_shape)` method.'.format(bad_example=bad_example, correct_example=correct_example))
        if method == 'add_metric':
            bad_example = "\n      def call(self, inputs, training=None):\n        if training:\n          metric = compute_metric(inputs)\n          self.add_metric(metric, name='my_metric', aggregation='mean')\n        return inputs\n      "
            correct_example = "\n      def call(self, inputs, training=None):\n        if training:\n          metric = compute_metric(inputs)\n        else:\n          metric = 0.\n        self.add_metric(metric, name='my_metric', aggregation='mean')\n        return inputs\n      "
        elif method == 'add_loss':
            bad_example = '\n      def call(self, inputs, training=None):\n        if training:\n          loss = compute_loss(inputs)\n          self.add_loss(loss)\n        return inputs\n      '
            correct_example = '\n      def call(self, inputs, training=None):\n        if training:\n          loss = compute_loss(inputs)\n        else:\n          loss = 0.\n        self.add_loss(loss)\n        return inputs\n      '
        else:
            bad_example = '\n      def call(self, inputs, training=None):\n        if training:\n          self.add_update(self.w.assign_add(1))\n        return inputs\n      '
            correct_example = '\n      def call(self, inputs, training=None):\n        if training:\n          increment = 1\n        else:\n          increment = 0\n        self.add_update(self.w.assign_add(increment))\n        return inputs\n      '
        raise RuntimeError('You are using the method `{method}` in a control flow branch in your layer, e.g.:\n{bad_example}\nThis is not currently supported. Please move your call to {method} out of the control flow branch, e.g.:\n{correct_example}\nYou can also resolve this by marking your layer as dynamic (eager-only) by passing `dynamic=True` to the layer constructor. Any kind of control flow is supported with dynamic layers. Note that using `dynamic=True` requires you to implement static shape inference in the `compute_output_shape(input_shape)` method.'.format(method=method, bad_example=bad_example, correct_example=correct_example))