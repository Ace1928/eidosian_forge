from tensorflow.python.ops import gen_collective_ops
def initialize_communicator(group_key, rank, group_size, communication_hint='auto', timeout_seconds=0):
    """Initializes a collective communicator.

  This creates a collective communicator, which represents membership to a
  collective group identified by the group_key. It should be called once per
  member of the group, and each member needs to be on a different device.
  It blocks until all members of the group run this op.

  Communicators of a group can only be initialized once. Trying to initialize
  communicators for an existing group key will result in an error.

  Args:
    group_key: an int32 `tf.Tensor` identifying the group.
    rank: an `tf.Tensor` specifying the rank of this device in the group. If
      specified, the rank is required to be unique in the group.
    group_size: an int32 `tf.Tensor`. The size of the group.
    communication_hint: preferred collective communication.  The implementation
      may fall back to another mechanism.  Options include `auto`, `ring`, and
      `nccl`.
    timeout_seconds: If set to a non zero, set a completion timeout to detect
      staleness. If the timer goes off, a DeadlineExceededError is raised. The
      timeout value in seconds. This feature is experimental.


  Returns:
    A resource `tf.Tensor`.
  """
    return gen_collective_ops.collective_initialize_communicator(group_key=group_key, rank=rank, group_size=group_size, communication_hint=communication_hint, timeout_seconds=timeout_seconds)