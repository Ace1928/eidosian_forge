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
def redirect_to_syslog(fds, server):
    log_cmd = 'logger'
    log_cmd_params = '-t "%s[%d]"' % (server, os.getpid())
    process = subprocess.Popen([log_cmd, log_cmd_params], stdin=subprocess.PIPE)
    for desc in fds:
        try:
            os.dup2(process.stdin.fileno(), desc)
        except OSError:
            pass