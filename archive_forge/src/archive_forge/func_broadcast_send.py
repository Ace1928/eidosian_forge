from tensorflow.python.ops import gen_collective_ops
def broadcast_send(t, shape, dtype, group_size, group_key, instance_key, communication_hint='auto', timeout=0):
    """Broadcasts one tensor to a group of others, across devices.

  Args:
    t: the tensor to be sent.
    shape: the shape of the tensor being sent, which must agree with t.
    dtype: the type of the tensor being sent, which must agree with t.
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
    An Op implementing the distributed broadcast send.

  Raises:
    ValueError: if any of the input parameter constraints are not met.

  Note that the shape and dtype arguments appear redundant since they
  should be obtainable from t.  The are two reasons for including
  them.  First, the shape and type of tensors passed via broadcast must
  be known ahead of time in their most specific form so that the receive
  side can allocate memory for the operation and shape/type inference can
  carry forward from there.  Including the same declarations on the
  send side clarifies a commitment already made.  Secondly, having nearly
  identical use syntax for send and receive sides may simplify tool-driven
  generation of broadcast.
  """
    if group_size <= 1:
        raise ValueError(f'Parameter `group_size` to broadcast_send must be at least 2. Received: {group_size}.')
    if t.shape != shape:
        raise ValueError(f'Shape of broadcast_send tensor `t` not equal to declared shape. Received {t.shape}, expected {shape}.')
    if t.dtype != dtype:
        raise ValueError(f'Type of broadcast_send tensor `t` not equal to declared type. Received {t.dtype}, expected {dtype}.')
    return gen_collective_ops.collective_bcast_send(t, shape=shape, group_size=group_size, group_key=group_key, instance_key=instance_key, communication_hint=communication_hint.lower(), timeout_seconds=timeout)