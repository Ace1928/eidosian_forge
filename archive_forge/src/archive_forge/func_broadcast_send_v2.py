from tensorflow.python.ops import gen_collective_ops
def broadcast_send_v2(t, group_size, group_key, instance_key, communication_hint='auto', timeout=0):
    """Broadcasts one tensor to a group of others, across devices.

  Args:
    t: the tensor to be sent.
    group_size: an int32 tensor.  One plus the number of receiving tensors, i.e.
        the total number of devices participating.  Each tensor must reside on a
        different device.
    group_key: an int32 tensor identifying the group of devices.
    instance_key: an int32 tensor identifying the participating group of Ops.
    communication_hint: preferred collective communication.  The implementation
      may fall back to another mechanism.  Options include `auto`, `ring`, and
      `nccl`.
    timeout: If set to a non zero, set a completion timeout to detect staleness.
      If the timer goes off, a DeadlineExceededError is raised.
      The timeout value in seconds. This feature is experimental.

  Returns:
    An Op implementing the distributed broadcast send.
  """
    return gen_collective_ops.collective_bcast_send_v2(t, group_size=group_size, group_key=group_key, instance_key=instance_key, communication_hint=communication_hint.lower(), timeout_seconds=timeout)