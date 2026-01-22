from pprint import pformat
from six import iteritems
import re
@last_heartbeat_time.setter
def last_heartbeat_time(self, last_heartbeat_time):
    """
        Sets the last_heartbeat_time of this V1NodeCondition.
        Last time we got an update on a given condition.

        :param last_heartbeat_time: The last_heartbeat_time of this
        V1NodeCondition.
        :type: datetime
        """
    self._last_heartbeat_time = last_heartbeat_time