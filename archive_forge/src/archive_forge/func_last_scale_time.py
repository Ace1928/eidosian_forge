from pprint import pformat
from six import iteritems
import re
@last_scale_time.setter
def last_scale_time(self, last_scale_time):
    """
        Sets the last_scale_time of this V2beta1HorizontalPodAutoscalerStatus.
        lastScaleTime is the last time the HorizontalPodAutoscaler scaled the
        number of pods, used by the autoscaler to control how often the number
        of pods is changed.

        :param last_scale_time: The last_scale_time of this
        V2beta1HorizontalPodAutoscalerStatus.
        :type: datetime
        """
    self._last_scale_time = last_scale_time