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
def del_watch(self, wd):
    """
        Remove watch entry associated to watch descriptor wd.

        @param wd: Watch descriptor.
        @type wd: int
        """
    try:
        del self._wmd[wd]
    except KeyError as err:
        log.error('Cannot delete unknown watch descriptor %s' % str(err))