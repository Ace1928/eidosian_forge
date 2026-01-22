from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.training import server_lib
def id_in_cluster(cluster_spec, task_type, task_id):
    """Returns a unique id for the task in the `task_type`'s cluster.

  It returns an id ranging from [0, `worker_count(task_type, task_id)`).

  Note: this function assumes that "evaluate" job is in its own cluster or its
  own partition of a cluster.

  Args:
    cluster_spec: a dict, `ClusterDef` or `ClusterSpec` object to be validated.
    task_type: string indicating the type of the task.
    task_id: the id of the `task_type` in this cluster.

  Returns:
    an int indicating the unique id.

  Throws:
    ValueError: if `task_type` is not "chief", "worker" or "evaluator".
  """
    _validate_cluster_spec(cluster_spec, task_type, task_id)
    cluster_spec = normalize_cluster_spec(cluster_spec).as_dict()
    if task_type == 'chief':
        return 0
    if task_type == 'worker':
        return task_id + len(cluster_spec.get('chief', []))
    if task_type == 'evaluator':
        return task_id
    raise ValueError('There is no id for task_type %r' % task_type)