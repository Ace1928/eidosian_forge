import errno
import io
import logging
import multiprocessing
import os
import pickle
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import time
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_concurrency import processutils
def test_check_exit_code_boolean(self):
    processutils.execute('/usr/bin/env', 'false', check_exit_code=False)
    self.assertRaises(processutils.ProcessExecutionError, processutils.execute, '/usr/bin/env', 'false', check_exit_code=True)