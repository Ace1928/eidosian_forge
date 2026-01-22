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
@mock.patch.object(multiprocessing, 'cpu_count', side_effect=NotImplementedError())
def test_get_worker_count_cpu_count_not_implemented(self, mock_cpu_count):
    self.assertEqual(1, processutils.get_worker_count())