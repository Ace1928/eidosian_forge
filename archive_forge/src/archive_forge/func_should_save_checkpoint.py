from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
def should_save_checkpoint():
    """Returns whether the current worker should save checkpoints.

  In multi-worker training, if saving checkpoint is requested by user, or needed
  for fault-tolerance, the cluster should save checkpoint but not necessarily
  every worker in the cluster should.

  TODO(rchao): Consider generalizing this util to be `should_save_file` as there
  can be other files to save such as summary.

  Returns:
      Whether this particular worker in the cluster should save checkpoints.
  """
    return dc_context.get_current_worker_context().should_checkpoint