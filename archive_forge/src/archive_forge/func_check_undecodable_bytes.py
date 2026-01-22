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
def check_undecodable_bytes(self, binary):
    out_bytes = b'out: ' + UNDECODABLE_BYTES
    err_bytes = b'err: ' + UNDECODABLE_BYTES
    conn = FakeSshConnection(0, out=out_bytes, err=err_bytes)
    out, err = processutils.ssh_execute(conn, 'ls', binary=binary)
    if not binary:
        self.assertEqual(os.fsdecode(out_bytes), out)
        self.assertEqual(os.fsdecode(err_bytes), err)
    else:
        self.assertEqual(out_bytes, out)
        self.assertEqual(err_bytes, err)