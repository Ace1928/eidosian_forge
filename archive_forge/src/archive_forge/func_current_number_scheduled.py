from pprint import pformat
from six import iteritems
import re
@current_number_scheduled.setter
def current_number_scheduled(self, current_number_scheduled):
    """
        Sets the current_number_scheduled of this V1beta2DaemonSetStatus.
        The number of nodes that are running at least 1 daemon pod and are
        supposed to run the daemon pod. More info:
        https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/

        :param current_number_scheduled: The current_number_scheduled of this
        V1beta2DaemonSetStatus.
        :type: int
        """
    if current_number_scheduled is None:
        raise ValueError('Invalid value for `current_number_scheduled`, must not be `None`')
    self._current_number_scheduled = current_number_scheduled