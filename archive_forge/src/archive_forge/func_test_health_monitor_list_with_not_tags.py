import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_health_monitor_list_with_not_tags(self):
    arglist = ['--not-tags', 'foo,bar']
    verifylist = [('not_tags', ['foo', 'bar'])]
    expected_attrs = {'not-tags': ['foo', 'bar']}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.health_monitor_list.assert_called_with(**expected_attrs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))