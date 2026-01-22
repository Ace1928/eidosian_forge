import collections
import contextlib
import copy
import logging
from oslo_utils import reflection
def can_trigger_notification(self, event_type):
    """Checks if the event can trigger a notification.

        :param event_type: event that needs to be verified
        :returns: whether the event can trigger a notification
        :rtype: boolean
        """
    if event_type in self._DISALLOWED_NOTIFICATION_EVENTS:
        return False
    else:
        return True