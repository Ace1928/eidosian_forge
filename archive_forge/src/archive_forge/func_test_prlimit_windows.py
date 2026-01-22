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
@mock.patch.object(os, 'name', 'nt')
@mock.patch.object(processutils.subprocess, 'Popen')
def test_prlimit_windows(self, mock_popen):
    prlimit = self.limit_address_space()
    mock_popen.return_value.communicate.return_value = None
    processutils.execute(*self.SIMPLE_PROGRAM, prlimit=prlimit, check_exit_code=False)
    mock_popen.assert_called_once_with(self.SIMPLE_PROGRAM, stdin=mock.ANY, stdout=mock.ANY, stderr=mock.ANY, close_fds=mock.ANY, preexec_fn=mock.ANY, shell=mock.ANY, cwd=mock.ANY, env=mock.ANY)