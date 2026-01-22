import argparse
import fcntl
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from oslo_config import cfg
from oslo_utils import units
from glance.common import config
from glance.i18n import _
def do_check_status(pid_file, server):
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as pidfile:
            pid = pidfile.read().strip()
        print(_('%(serv)s (pid %(pid)s) is running...') % {'serv': server, 'pid': pid})
    else:
        print(_('%s is stopped') % server)