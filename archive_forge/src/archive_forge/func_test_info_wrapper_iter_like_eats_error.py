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
def test_info_wrapper_iter_like_eats_error(self):
    fake_fmt = mock.create_autospec(format_inspector.get_inspector('raw'))
    wrapper = format_inspector.InfoWrapper(iter([b'123', b'456']), fake_fmt)
    fake_fmt.eat_chunk.side_effect = Exception('fail')
    data = b''
    for chunk in wrapper:
        data += chunk
    self.assertEqual(b'123456', data)
    fake_fmt.eat_chunk.assert_called_once_with(b'123')