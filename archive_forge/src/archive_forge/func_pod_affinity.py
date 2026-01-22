from pprint import pformat
from six import iteritems
import re
@pod_affinity.setter
def pod_affinity(self, pod_affinity):
    """
        Sets the pod_affinity of this V1Affinity.
        Describes pod affinity scheduling rules (e.g. co-locate this pod in the
        same node, zone, etc. as some other pod(s)).

        :param pod_affinity: The pod_affinity of this V1Affinity.
        :type: V1PodAffinity
        """
    self._pod_affinity = pod_affinity