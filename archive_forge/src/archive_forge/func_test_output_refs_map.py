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
def test_output_refs_map(self):
    mock_members = self.patchobject(grouputils, 'get_members')
    members = [mock.MagicMock(), mock.MagicMock()]
    members[0].name = 'resource-1-name'
    members[0].resource_id = 'resource-1-id'
    members[1].name = 'resource-2-name'
    members[1].resource_id = 'resource-2-id'
    mock_members.return_value = members
    found = self.group.FnGetAtt('refs_map')
    expected = {'resource-1-name': 'resource-1-id', 'resource-2-name': 'resource-2-id'}
    self.assertEqual(expected, found)