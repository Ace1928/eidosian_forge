from pprint import pformat
from six import iteritems
import re
@current_healthy.setter
def current_healthy(self, current_healthy):
    """
        Sets the current_healthy of this V1beta1PodDisruptionBudgetStatus.
        current number of healthy pods

        :param current_healthy: The current_healthy of this
        V1beta1PodDisruptionBudgetStatus.
        :type: int
        """
    if current_healthy is None:
        raise ValueError('Invalid value for `current_healthy`, must not be `None`')
    self._current_healthy = current_healthy