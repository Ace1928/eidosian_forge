from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.get_global_step'])
def get_global_step(graph=None):
    """Get the global step tensor.

  The global step tensor must be an integer variable. We first try to find it
  in the collection `GLOBAL_STEP`, or by name `global_step:0`.

  Args:
    graph: The graph to find the global step in. If missing, use default graph.

  Returns:
    The global step variable, or `None` if none was found.

  Raises:
    TypeError: If the global step tensor has a non-integer type, or if it is not
      a `Variable`.

  @compatibility(TF2)
  With the deprecation of global graphs, TF no longer tracks variables in
  collections. In other words, there are no global variables in TF2. Thus, the
  global step functions have been removed  (`get_or_create_global_step`,
  `create_global_step`, `get_global_step`) . You have two options for migrating:

  1. Create a Keras optimizer, which generates an `iterations` variable. This
     variable is automatically incremented when calling `apply_gradients`.
  2. Manually create and increment a `tf.Variable`.

  Below is an example of migrating away from using a global step to using a
  Keras optimizer:

  Define a dummy model and loss:

  >>> def compute_loss(x):
  ...   v = tf.Variable(3.0)
  ...   y = x * v
  ...   loss = x * 5 - x * v
  ...   return loss, [v]

  Before migrating:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   x = tf.compat.v1.placeholder(tf.float32, [])
  ...   loss, var_list = compute_loss(x)
  ...   global_step = tf.compat.v1.train.get_or_create_global_step()
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
  ...   train_op = optimizer.minimize(loss, global_step, var_list)
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run(global_init)
  >>> print("before training:", sess.run(global_step))
  before training: 0
  >>> sess.run(train_op, feed_dict={x: 3})
  >>> print("after training:", sess.run(global_step))
  after training: 1

  Using `get_global_step`:

  >>> with g.as_default():
  ...   print(sess.run(tf.compat.v1.train.get_global_step()))
  1

  Migrating to a Keras optimizer:

  >>> optimizer = tf.keras.optimizers.SGD(.01)
  >>> print("before training:", optimizer.iterations.numpy())
  before training: 0
  >>> with tf.GradientTape() as tape:
  ...   loss, var_list = compute_loss(3)
  ...   grads = tape.gradient(loss, var_list)
  ...   optimizer.apply_gradients(zip(grads, var_list))
  >>> print("after training:", optimizer.iterations.numpy())
  after training: 1

  @end_compatibility
  """
    graph = graph or ops.get_default_graph()
    global_step_tensor = None
    global_step_tensors = graph.get_collection(ops.GraphKeys.GLOBAL_STEP)
    if len(global_step_tensors) == 1:
        global_step_tensor = global_step_tensors[0]
    elif not global_step_tensors:
        try:
            global_step_tensor = graph.get_tensor_by_name('global_step:0')
        except KeyError:
            return None
    else:
        logging.error('Multiple tensors in global_step collection.')
        return None
    assert_global_step(global_step_tensor)
    return global_step_tensor