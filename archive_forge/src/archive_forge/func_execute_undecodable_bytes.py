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
def execute_undecodable_bytes(self, out_bytes, err_bytes, exitcode=0, binary=False):
    code = ';'.join(('import sys', 'sys.stdout.buffer.write(%a)' % out_bytes, 'sys.stdout.flush()', 'sys.stderr.buffer.write(%a)' % err_bytes, 'sys.stderr.flush()', 'sys.exit(%s)' % exitcode))
    return processutils.execute(sys.executable, '-c', code, binary=binary)