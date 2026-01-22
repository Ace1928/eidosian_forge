from pprint import pformat
from six import iteritems
import re
@expected_pods.setter
def expected_pods(self, expected_pods):
    """
        Sets the expected_pods of this V1beta1PodDisruptionBudgetStatus.
        total number of pods counted by this disruption budget

        :param expected_pods: The expected_pods of this
        V1beta1PodDisruptionBudgetStatus.
        :type: int
        """
    if expected_pods is None:
        raise ValueError('Invalid value for `expected_pods`, must not be `None`')
    self._expected_pods = expected_pods