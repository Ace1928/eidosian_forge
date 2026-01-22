import io
import os
import re
import struct
import subprocess
import tempfile
from unittest import mock
from oslo_utils import units
from glance.common import format_inspector
from glance.tests import utils as test_utils
def test_info_wrapper_file_like(self):
    data = b''.join((chr(x).encode() for x in range(ord('A'), ord('z'))))
    wrapper = self._get_wrapper(data)
    read_data = b''
    while True:
        chunk = wrapper.read(8)
        if not chunk:
            break
        read_data += chunk
    self.assertEqual(data, read_data)