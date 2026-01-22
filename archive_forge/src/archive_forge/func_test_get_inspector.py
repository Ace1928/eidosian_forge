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
def test_get_inspector(self):
    self.assertEqual(format_inspector.QcowInspector, format_inspector.get_inspector('qcow2'))
    self.assertIsNone(format_inspector.get_inspector('foo'))