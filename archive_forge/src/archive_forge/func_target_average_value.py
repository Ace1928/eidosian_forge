from pprint import pformat
from six import iteritems
import re
@target_average_value.setter
def target_average_value(self, target_average_value):
    """
        Sets the target_average_value of this V2beta1ExternalMetricSource.
        targetAverageValue is the target per-pod value of global metric (as a
        quantity). Mutually exclusive with TargetValue.

        :param target_average_value: The target_average_value of this
        V2beta1ExternalMetricSource.
        :type: str
        """
    self._target_average_value = target_average_value