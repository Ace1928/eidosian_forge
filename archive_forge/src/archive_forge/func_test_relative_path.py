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
def test_relative_path(self):
    prlimit = self.limit_address_space()
    program = sys.executable
    env = dict(os.environ)
    env['PATH'] = os.path.dirname(program)
    args = [os.path.basename(program), '-c', 'pass']
    processutils.execute(*args, prlimit=prlimit, env_variables=env)