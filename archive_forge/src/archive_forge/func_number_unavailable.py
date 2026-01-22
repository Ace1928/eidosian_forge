from pprint import pformat
from six import iteritems
import re
@number_unavailable.setter
def number_unavailable(self, number_unavailable):
    """
        Sets the number_unavailable of this V1beta2DaemonSetStatus.
        The number of nodes that should be running the daemon pod and have none
        of the daemon pod running and available (ready for at least
        spec.minReadySeconds)

        :param number_unavailable: The number_unavailable of this
        V1beta2DaemonSetStatus.
        :type: int
        """
    self._number_unavailable = number_unavailable