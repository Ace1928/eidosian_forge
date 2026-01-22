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
def str_errno(self):
    code = self.get_errno()
    if code is None:
        return 'Errno: no errno support'
    return 'Errno=%s (%s)' % (os.strerror(code), errno.errorcode[code])