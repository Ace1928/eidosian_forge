from pprint import pformat
from six import iteritems
import re
@min_ready_seconds.setter
def min_ready_seconds(self, min_ready_seconds):
    """
        Sets the min_ready_seconds of this V1beta2ReplicaSetSpec.
        Minimum number of seconds for which a newly created pod should be ready
        without any of its container crashing, for it to be considered
        available. Defaults to 0 (pod will be considered available as soon as it
        is ready)

        :param min_ready_seconds: The min_ready_seconds of this
        V1beta2ReplicaSetSpec.
        :type: int
        """
    self._min_ready_seconds = min_ready_seconds