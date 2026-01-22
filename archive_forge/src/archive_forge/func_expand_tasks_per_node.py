import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def expand_tasks_per_node(tasks_per_node):
    """Expands the tasks per node expression from SLURM.

  The order is preserved so it can be matched to the hostlist
  Input: '3(x2),2,1'
  Output: [3, 3, 2, 1]
  """
    result = []
    try:
        for part in tasks_per_node.split(','):
            m = re.match('(\\d+)(\\(x(\\d+)\\))?$', part)
            assert m is not None
            num_tasks = int(m.group(1))
            num_repetitions = int(m.group(3) or 1)
            result.extend([num_tasks] * num_repetitions)
    except Exception as e:
        raise ValueError('Invalid tasks-per-node list format "%s": %s' % (tasks_per_node, e))
    return result