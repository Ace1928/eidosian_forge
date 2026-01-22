from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
def has_worker_context():
    """Returns whether a worker context has been entered."""
    return dc_context.get_current_worker_context() is not None