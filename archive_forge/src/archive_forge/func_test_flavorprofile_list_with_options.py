import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import flavorprofile
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_flavorprofile_list_with_options(self):
    arglist = ['--name', 'flavorprofile1']
    verifylist = [('name', 'flavorprofile1')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.api_mock.flavorprofile_list.assert_called_with(name='flavorprofile1')
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))