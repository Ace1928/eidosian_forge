from pprint import pformat
from six import iteritems
import re
@last_observed_time.setter
def last_observed_time(self, last_observed_time):
    """
        Sets the last_observed_time of this V1beta1EventSeries.
        Time when last Event from the series was seen before last heartbeat.

        :param last_observed_time: The last_observed_time of this
        V1beta1EventSeries.
        :type: datetime
        """
    if last_observed_time is None:
        raise ValueError('Invalid value for `last_observed_time`, must not be `None`')
    self._last_observed_time = last_observed_time