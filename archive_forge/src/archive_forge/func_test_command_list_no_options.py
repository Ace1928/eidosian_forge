from unittest import mock
from openstackclient.common import module as osc_module
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
def test_command_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    collist = ('Command Group', 'Commands')
    self.assertEqual(collist, columns)
    datalist = (('openstack.common', 'limits show\nextension list'),)
    self.assertEqual(datalist, tuple(data))