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
def test_get_template_contents_parse_error(self):
    with tempfile.NamedTemporaryFile() as tmpl_file:
        tmpl = b'{"foo": "bar"'
        tmpl_file.write(tmpl)
        tmpl_file.flush()
        ex = self.assertRaises(exc.CommandError, template_utils.get_template_contents, tmpl_file.name)
        self.assertThat(str(ex), matchers.MatchesRegex('Error parsing template file://%s ' % tmpl_file.name))