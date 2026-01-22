from pprint import pformat
from six import iteritems
import re
@initial_delay_seconds.setter
def initial_delay_seconds(self, initial_delay_seconds):
    """
        Sets the initial_delay_seconds of this V1Probe.
        Number of seconds after the container has started before liveness probes
        are initiated. More info:
        https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle#container-probes

        :param initial_delay_seconds: The initial_delay_seconds of this V1Probe.
        :type: int
        """
    self._initial_delay_seconds = initial_delay_seconds