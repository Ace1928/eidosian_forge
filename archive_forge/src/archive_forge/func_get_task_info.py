import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def get_task_info(self):
    """Returns job name and task_id for the process which calls this.

    This returns the job name and task index for the process which calls this
    function according to its rank and cluster specification. The job name and
    task index are set after a cluster is constructed by cluster_spec otherwise
    defaults to None.

    Returns:
      A string specifying job name the process belongs to and an integer
        specifying the task index the process belongs to in that job.
    """
    return (self.task_type, self.task_id)