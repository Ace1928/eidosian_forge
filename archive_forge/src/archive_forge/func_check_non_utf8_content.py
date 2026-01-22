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
def check_non_utf8_content(self, filename, content):
    base_url = 'file:///tmp'
    url = '%s/%s' % (base_url, filename)
    template = {'resources': {'one_init': {'type': 'OS::Heat::CloudConfig', 'properties': {'cloud_config': {'write_files': [{'path': '/tmp/%s' % filename, 'content': {'get_file': url}, 'encoding': 'b64'}]}}}}}
    with mock.patch('urllib.request.urlopen') as mock_url:
        raw_content = base64.decode_as_bytes(content)
        response = io.BytesIO(raw_content)
        mock_url.return_value = response
        files = {}
        template_utils.resolve_template_get_files(template, files, base_url)
        self.assertEqual({url: content}, files)
        mock_url.assert_called_with(url)