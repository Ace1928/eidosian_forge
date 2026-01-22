from tensorflow.python.ops import gen_collective_ops
def broadcast_recv(shape, dtype, group_size, group_key, instance_key, communication_hint='auto', timeout=0):
    """Receives a broadcasts tensor, across devices.

  Args:
    shape: Shape of the tensor to be received.
    dtype: Type of the tensor to be received.
    group_size: one plus the number of receiving tensors, i.e. the total
      number of devices participating.  Each tensor must reside on a
      different device.
    group_key: an integer identifying the group of devices.
    instance_key: an integer identifying the participating group of Ops.
    communication_hint: preferred collective communication.  The implementation
      may fall back to another mechanism.  Options include `auto`, `ring`, and
      `nccl`.
    timeout: If set to a non zero, set a completion timeout to detect staleness.
      If the timer goes off, a DeadlineExceededError is raised.
      The timeout value in seconds. This feature is experimental.

  Returns:
    An Op implementing the broadcast receive.

  Raises:
    ValueError: if any of the input parameter constraints are not met.
  """
    if group_size <= 1:
        raise ValueError(f'Parameter `group_size` to broadcast_send must be at least 2. Received: {group_size}.')
    return gen_collective_ops.collective_bcast_recv(shape=shape, T=dtype, group_size=group_size, group_key=group_key, instance_key=instance_key, communication_hint=communication_hint.lower(), timeout_seconds=timeout)