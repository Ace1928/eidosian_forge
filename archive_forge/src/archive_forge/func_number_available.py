from pprint import pformat
from six import iteritems
import re
@number_available.setter
def number_available(self, number_available):
    """
        Sets the number_available of this V1beta2DaemonSetStatus.
        The number of nodes that should be running the daemon pod and have one
        or more of the daemon pod running and available (ready for at least
        spec.minReadySeconds)

        :param number_available: The number_available of this
        V1beta2DaemonSetStatus.
        :type: int
        """
    self._number_available = number_available