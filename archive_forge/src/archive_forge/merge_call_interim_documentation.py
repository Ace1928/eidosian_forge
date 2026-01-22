from tensorflow.python.distribute import distribute_lib
from tensorflow.python.util.tf_export import tf_export
Maybe invoke `fn` via `merge_call` which may or may not be fulfilled.

  The caller of this utility function requests to invoke `fn` via `merge_call`
  at `tf.distribute.Strategy`'s best efforts. It is `tf.distribute`'s internal
  whether the request is honored, depending on the `Strategy`. See
  `tf.distribute.ReplicaContext.merge_call()` for more information.

  This is an interim API which is subject to removal and does not guarantee
  backward-compatibility.

  Args:
    fn: the function to be invoked.
    strategy: the `tf.distribute.Strategy` to call `fn` with.
    *args: the positional arguments to be passed in to `fn`.
    **kwargs: the keyword arguments to be passed in to `fn`.

  Returns:
    The return value of the `fn` call.
  