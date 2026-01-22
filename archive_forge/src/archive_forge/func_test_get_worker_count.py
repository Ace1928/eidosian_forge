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
@mock.patch.object(multiprocessing, 'cpu_count', return_value=8)
def test_get_worker_count(self, mock_cpu_count):
    self.assertEqual(8, processutils.get_worker_count())