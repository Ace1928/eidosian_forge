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
def test_vmdk_bad_descriptor_mem_limit_stream_optimized(self):
    self._test_vmdk_bad_descriptor_mem_limit(subformat='streamOptimized')