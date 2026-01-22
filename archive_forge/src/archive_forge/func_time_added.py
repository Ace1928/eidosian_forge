from pprint import pformat
from six import iteritems
import re
@time_added.setter
def time_added(self, time_added):
    """
        Sets the time_added of this V1Taint.
        TimeAdded represents the time at which the taint was added. It is only
        written for NoExecute taints.

        :param time_added: The time_added of this V1Taint.
        :type: datetime
        """
    self._time_added = time_added