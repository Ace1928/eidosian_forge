from pprint import pformat
from six import iteritems
import re
@desired_replicas.setter
def desired_replicas(self, desired_replicas):
    """
        Sets the desired_replicas of this V2beta1HorizontalPodAutoscalerStatus.
        desiredReplicas is the desired number of replicas of pods managed by
        this autoscaler, as last calculated by the autoscaler.

        :param desired_replicas: The desired_replicas of this
        V2beta1HorizontalPodAutoscalerStatus.
        :type: int
        """
    if desired_replicas is None:
        raise ValueError('Invalid value for `desired_replicas`, must not be `None`')
    self._desired_replicas = desired_replicas