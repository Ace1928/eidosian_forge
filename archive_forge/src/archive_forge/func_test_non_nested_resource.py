from unittest import mock
from heat.common import grouputils
from heat.common import identifier
from heat.common import template_format
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_non_nested_resource(self):
    group = mock.Mock()
    group.nested_identifier.return_value = None
    group.nested.return_value = None
    self.assertEqual(0, grouputils.get_size(group))
    self.assertEqual([], grouputils.get_members(group))
    self.assertEqual([], grouputils.get_member_refids(group))
    self.assertEqual([], grouputils.get_member_names(group))