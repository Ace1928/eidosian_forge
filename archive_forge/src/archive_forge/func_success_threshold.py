from pprint import pformat
from six import iteritems
import re
@success_threshold.setter
def success_threshold(self, success_threshold):
    """
        Sets the success_threshold of this V1Probe.
        Minimum consecutive successes for the probe to be considered successful
        after having failed. Defaults to 1. Must be 1 for liveness. Minimum
        value is 1.

        :param success_threshold: The success_threshold of this V1Probe.
        :type: int
        """
    self._success_threshold = success_threshold