import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
def process_IN_IGNORED(self, raw_event):
    """
        The watch descriptor raised by this event is now ignored (forever),
        it can be safely deleted from the watch manager dictionary.
        After this event we can be sure that neither the event queue nor
        the system will raise an event associated to this wd again.
        """
    event_ = self.process_default(raw_event)
    self._watch_manager.del_watch(raw_event.wd)
    return event_