import io
from oslo_log.cmds import convert_json
from oslo_serialization import jsonutils
from oslotest import base as test_base
def test_console_format_trivial(self):
    lines = self._lines(TRIVIAL_RECORD)
    self.assertEqual(['pre msg'], lines)