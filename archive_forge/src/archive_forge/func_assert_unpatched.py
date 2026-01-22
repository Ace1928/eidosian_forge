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
def assert_unpatched(self):
    if eventlet and eventlet.patcher.is_monkey_patched('socket'):
        self.fail('Standard library should not be patched by eventlet for this test')