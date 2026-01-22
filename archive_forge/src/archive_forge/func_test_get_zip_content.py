import json
import tempfile
from unittest import mock
import io
from oslo_serialization import base64
import testtools
from testtools import matchers
from urllib import error
import yaml
from heatclient.common import template_utils
from heatclient.common import utils
from heatclient import exc
def test_get_zip_content(self):
    filename = 'heat.zip'
    content = b'UEsDBAoAAAAAAEZZWkRbOAuBBQAAAAUAAAAIABwAaGVhdC50eHRVVAkAAxRbDVNYht9SdXgLAAEE\n6AMAAATpAwAAaGVhdApQSwECHgMKAAAAAABGWVpEWzgLgQUAAAAFAAAACAAYAAAAAAABAAAApIEA\nAAAAaGVhdC50eHRVVAUAAxRbDVN1eAsAAQToAwAABOkDAABQSwUGAAAAAAEAAQBOAAAARwAAAAAA\n'
    self.assertIn(b'\x00', base64.decode_as_bytes(content))
    decoded_content = base64.decode_as_bytes(content)
    self.assertRaises(UnicodeDecodeError, decoded_content.decode)
    self.check_non_utf8_content(filename=filename, content=content)