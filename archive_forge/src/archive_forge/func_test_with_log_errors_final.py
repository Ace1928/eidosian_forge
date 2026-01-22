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
def test_with_log_errors_final(self):
    self._test_and_check(log_errors=processutils.LOG_FINAL_ERROR, attempts=None)