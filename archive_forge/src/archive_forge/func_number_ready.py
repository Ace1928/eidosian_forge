from pprint import pformat
from six import iteritems
import re
@number_ready.setter
def number_ready(self, number_ready):
    """
        Sets the number_ready of this V1beta2DaemonSetStatus.
        The number of nodes that should be running the daemon pod and have one
        or more of the daemon pod running and ready.

        :param number_ready: The number_ready of this V1beta2DaemonSetStatus.
        :type: int
        """
    if number_ready is None:
        raise ValueError('Invalid value for `number_ready`, must not be `None`')
    self._number_ready = number_ready