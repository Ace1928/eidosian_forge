import json
import os
import tensorflow.compat.v2 as tf
def configure_cluster(worker_hosts=None, task_index=-1):
    """Set multi-worker cluster spec in TF_CONFIG environment variable.

    Args:
      worker_hosts: comma-separated list of worker ip:port pairs.

    Returns:
      Number of workers in the cluster.
    """
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    if tf_config:
        num_workers = len(tf_config['cluster'].get('chief', [])) + len(tf_config['cluster'].get('worker', []))
    elif worker_hosts:
        workers = worker_hosts.split(',')
        num_workers = len(workers)
        if num_workers > 1 and task_index < 0:
            raise ValueError('Must specify task_index when number of workers > 1')
        task_index = 0 if num_workers == 1 else task_index
        os.environ['TF_CONFIG'] = json.dumps({'cluster': {'worker': workers}, 'task': {'type': 'worker', 'index': task_index}})
    else:
        num_workers = 1
    return num_workers