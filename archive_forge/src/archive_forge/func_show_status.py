import atexit
import errno
import logging
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from paste.deploy import loadapp, loadserver
from paste.script.command import Command, BadCommand
def show_status(self):
    pid_file = self.options.pid_file or 'paster.pid'
    if not os.path.exists(pid_file):
        print('No PID file %s' % pid_file)
        return 1
    pid = read_pidfile(pid_file)
    if not pid:
        print('No PID in file %s' % pid_file)
        return 1
    pid = live_pidfile(pid_file)
    if not pid:
        print('PID %s in %s is not running' % (pid, pid_file))
        return 1
    print('Server running in PID %s' % pid)
    return 0