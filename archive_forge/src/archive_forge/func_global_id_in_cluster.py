from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@property
def global_id_in_cluster(self):
    """The global id in the training cluster.

    All global ids in the training cluster are assigned from an increasing
    sequence of consecutive integers. The first id is 0.

    Note: Task id (the property field `task_id`) is tracking the index of the
    node among all nodes with the SAME task type. For example, given the cluster
    definition as follows:

    ```
      cluster = {'chief': ['host0:2222'],
                 'ps': ['host1:2222', 'host2:2222'],
                 'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
    ```

    Nodes with task type `worker` can have id 0, 1, 2.  Nodes with task type
    `ps` can have id, 0, 1. So, `task_id` is not unique, but the pair
    (`task_type`, `task_id`) can uniquely determine a node in the cluster.

    Global id, i.e., this field, is tracking the index of the node among ALL
    nodes in the cluster. It is uniquely assigned.  For example, for the cluster
    spec given above, the global ids are assigned as:
    ```
      task_type  | task_id  |  global_id
      --------------------------------
      chief      | 0        |  0
      worker     | 0        |  1
      worker     | 1        |  2
      worker     | 2        |  3
      ps         | 0        |  4
      ps         | 1        |  5
    ```

    Returns:
      An integer id.
    """
    return self._global_id_in_cluster