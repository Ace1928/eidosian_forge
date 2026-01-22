from pprint import pformat
from six import iteritems
import re
@max_replicas.setter
def max_replicas(self, max_replicas):
    """
        Sets the max_replicas of this V2beta2HorizontalPodAutoscalerSpec.
        maxReplicas is the upper limit for the number of replicas to which the
        autoscaler can scale up. It cannot be less that minReplicas.

        :param max_replicas: The max_replicas of this
        V2beta2HorizontalPodAutoscalerSpec.
        :type: int
        """
    if max_replicas is None:
        raise ValueError('Invalid value for `max_replicas`, must not be `None`')
    self._max_replicas = max_replicas