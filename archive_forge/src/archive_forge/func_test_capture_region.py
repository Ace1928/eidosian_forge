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
def test_capture_region(self):
    for block_size in (1, 3, 7, 13, 32, 64):
        self._test_capture_region_bs(block_size)