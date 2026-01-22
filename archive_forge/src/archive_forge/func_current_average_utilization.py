from pprint import pformat
from six import iteritems
import re
@current_average_utilization.setter
def current_average_utilization(self, current_average_utilization):
    """
        Sets the current_average_utilization of this
        V2beta1ResourceMetricStatus.
        currentAverageUtilization is the current value of the average of the
        resource metric across all relevant pods, represented as a percentage of
        the requested value of the resource for the pods.  It will only be
        present if `targetAverageValue` was set in the corresponding metric
        specification.

        :param current_average_utilization: The current_average_utilization of
        this V2beta1ResourceMetricStatus.
        :type: int
        """
    self._current_average_utilization = current_average_utilization