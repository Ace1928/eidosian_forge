import contextlib
import io
import logging
import os
import pwd
import shutil
import signal
import sys
import threading
import time
from unittest import mock
import fixtures
import testtools
from testtools import content
from oslo_rootwrap import client
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
from oslo_rootwrap.tests import run_daemon
def test_daemon_timeout(self):
    self.execute(['echo'])
    with mock.patch.object(self.client, '_restart') as restart:
        time.sleep(15)
        self.execute(['echo'])
        restart.assert_called_once()