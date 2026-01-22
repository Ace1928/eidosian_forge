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
def test_get_template_contents_file_none(self):
    ex = self.assertRaises(exc.CommandError, template_utils.get_template_contents)
    self.assertEqual('Need to specify exactly one of [--template-file, --template-url or --template-object] or --existing', str(ex))