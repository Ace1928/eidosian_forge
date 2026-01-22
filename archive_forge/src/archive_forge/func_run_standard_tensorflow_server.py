import copy
import json
import os
import threading
import time
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
def run_standard_tensorflow_server(session_config=None):
    """Starts a standard TensorFlow server.

  This method parses configurations from "TF_CONFIG" environment variable and
  starts a TensorFlow server. The "TF_CONFIG" is typically a json string and
  must have information of the cluster and the role of the server in the
  cluster. One example is:

  TF_CONFIG='{
      "cluster": {
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "worker", "index": 1}
  }'

  This "TF_CONFIG" specifies there are 3 workers and 2 ps tasks in the cluster
  and the current role is worker 1.

  Valid task types are "chief", "worker", "ps" and "evaluator" and you can have
  at most one "chief" and at most one "evaluator".

  An optional key-value can be specified is "rpc_layer". The default value is
  "grpc".

  Args:
    session_config: an optional `tf.compat.v1.ConfigProto` object. Users can
      pass in the session config object to configure server-local devices.

  Returns:
    a `tf.distribute.Server` object which has already been started.

  Raises:
    ValueError: if the "TF_CONFIG" environment is not complete.
  """
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    if 'cluster' not in tf_config:
        raise ValueError('"cluster" is not found in TF_CONFIG.')
    cluster_spec = multi_worker_util.normalize_cluster_spec(tf_config['cluster'])
    if 'task' not in tf_config:
        raise ValueError('"task" is not found in TF_CONFIG.')
    task_env = tf_config['task']
    if 'type' not in task_env:
        raise ValueError('"task_type" is not found in the `task` part of TF_CONFIG.')
    task_type = task_env['type']
    task_id = int(task_env.get('index', 0))
    rpc_layer = tf_config.get('rpc_layer', 'grpc')
    session_config = session_config or config_pb2.ConfigProto()
    if 'chief' in cluster_spec.jobs:
        session_config.experimental.collective_group_leader = '/job:chief/replica:0/task:0'
    else:
        if 'worker' not in cluster_spec.jobs:
            raise ValueError('You must have `chief` or `worker` jobs in the `cluster_spec`.')
        session_config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
    server = _run_std_server(cluster_spec=cluster_spec, task_type=task_type, task_id=task_id, session_config=session_config, rpc_layer=rpc_layer)
    server.start()
    return server