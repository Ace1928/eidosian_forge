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
def test_vmdk_invalid_type(self):
    fmt = format_inspector.get_inspector('vmdk')()
    wrapper = format_inspector.InfoWrapper(open(__file__, 'rb'), fmt)
    while True:
        chunk = wrapper.read(32)
        if not chunk:
            break
    wrapper.close()
    fake_rgn = mock.MagicMock()
    fake_rgn.complete = True
    fake_rgn.data = b'foocreateType="someunknownformat"bar'
    with mock.patch.object(fmt, 'has_region', return_value=True):
        with mock.patch.object(fmt, 'region', return_value=fake_rgn):
            self.assertEqual(0, fmt.virtual_size)