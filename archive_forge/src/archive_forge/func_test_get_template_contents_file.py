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
def test_get_template_contents_file(self):
    with tempfile.NamedTemporaryFile() as tmpl_file:
        tmpl = b'{"AWSTemplateFormatVersion" : "2010-09-09", "foo": "bar"}'
        tmpl_file.write(tmpl)
        tmpl_file.flush()
        files, tmpl_parsed = template_utils.get_template_contents(tmpl_file.name)
        self.assertEqual({'AWSTemplateFormatVersion': '2010-09-09', 'foo': 'bar'}, tmpl_parsed)
        self.assertEqual({}, files)