from __future__ import absolute_import, division, print_function
import errno
import json
import shlex
import shutil
import os
import subprocess
import sys
import traceback
import signal
import time
import syslog
import multiprocessing
from ansible.module_utils.common.text.converters import to_text, to_bytes
def daemonize_self():
    try:
        pid = os.fork()
        if pid > 0:
            end()
    except OSError:
        e = sys.exc_info()[1]
        end({'msg': 'fork #1 failed: %d (%s)\n' % (e.errno, e.strerror), 'failed': True}, 1)
    os.setsid()
    os.umask(int('022', 8))
    try:
        pid = os.fork()
        if pid > 0:
            end()
    except OSError:
        e = sys.exc_info()[1]
        end({'msg': 'fork #2 failed: %d (%s)\n' % (e.errno, e.strerror), 'failed': True}, 1)
    dev_null = open('/dev/null', 'w')
    os.dup2(dev_null.fileno(), sys.stdin.fileno())
    os.dup2(dev_null.fileno(), sys.stdout.fileno())
    os.dup2(dev_null.fileno(), sys.stderr.fileno())