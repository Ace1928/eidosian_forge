import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
def timer_done(self, timer):
    """Remove a timer from the group.

        :param timer: The timer object returned from :func:`add_timer` or its
                      analogues.
        """
    self.timers.remove(timer)