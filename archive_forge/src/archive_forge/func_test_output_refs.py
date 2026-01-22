import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_output_refs(self):
    mock_get = self.patchobject(grouputils, 'get_member_refids')
    mock_get.return_value = ['resource-1', 'resource-2']
    found = self.group.FnGetAtt('refs')
    expected = ['resource-1', 'resource-2']
    self.assertEqual(expected, found)
    mock_get.assert_called_once_with(self.group)