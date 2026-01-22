import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
def thread_done(self, thread):
    """Remove a completed thread from the group.

        This method is automatically called on completion of a thread in the
        group, and should not be called explicitly.
        """
    self.threads.remove(thread)