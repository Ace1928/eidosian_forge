import io
from oslo_config import fixture as config
from oslotest import base as test_base
import webob
from oslo_middleware import sizelimit
def test_read_default_value(self):
    BYTES = 1024
    data_str = '*' * BYTES
    data = io.StringIO(data_str)
    reader = sizelimit.LimitingReader(data, BYTES)
    res = reader.read()
    self.assertEqual(data_str, res)