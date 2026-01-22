from pprint import pformat
from six import iteritems
import re
@ready_replicas.setter
def ready_replicas(self, ready_replicas):
    """
        Sets the ready_replicas of this V1ReplicaSetStatus.
        The number of ready replicas for this replica set.

        :param ready_replicas: The ready_replicas of this V1ReplicaSetStatus.
        :type: int
        """
    self._ready_replicas = ready_replicas