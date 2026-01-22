import io
from oslo_log.cmds import convert_json
from oslo_serialization import jsonutils
from oslotest import base as test_base
def test_console_format_debug_level_key(self):
    lines = self._lines(DEBUG_LEVEL_KEY_RECORD, level_key='level')
    self.assertEqual(['pre msg'], lines)