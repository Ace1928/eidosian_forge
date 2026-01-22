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
@gated_by(CONF.await_child)
def await_child(pid, await_time):
    bail_time = time.time() + await_time
    while time.time() < bail_time:
        reported_pid, status = os.waitpid(pid, os.WNOHANG)
        if reported_pid == pid:
            global exitcode
            exitcode = os.WEXITSTATUS(status)
            break
        time.sleep(0.05)