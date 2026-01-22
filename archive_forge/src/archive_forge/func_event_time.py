from pprint import pformat
from six import iteritems
import re
@event_time.setter
def event_time(self, event_time):
    """
        Sets the event_time of this V1beta1Event.
        Required. Time when this Event was first observed.

        :param event_time: The event_time of this V1beta1Event.
        :type: datetime
        """
    if event_time is None:
        raise ValueError('Invalid value for `event_time`, must not be `None`')
    self._event_time = event_time