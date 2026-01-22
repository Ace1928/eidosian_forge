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
def compatibility_mode():
    """
    Use this function to turn on the compatibility mode. The compatibility
    mode is used to improve compatibility with Pyinotify 0.7.1 (or older)
    programs. The compatibility mode provides additional variables 'is_dir',
    'event_name', 'EventsCodes.IN_*' and 'EventsCodes.ALL_EVENTS' as
    Pyinotify 0.7.1 provided. Do not call this function from new programs!!
    Especially if there are developped for Pyinotify >= 0.8.x.
    """
    setattr(EventsCodes, 'ALL_EVENTS', ALL_EVENTS)
    for evname in globals():
        if evname.startswith('IN_'):
            setattr(EventsCodes, evname, globals()[evname])
    global COMPATIBILITY_MODE
    COMPATIBILITY_MODE = True