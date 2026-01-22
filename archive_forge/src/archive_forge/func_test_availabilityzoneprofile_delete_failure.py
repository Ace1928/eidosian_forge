import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzoneprofile
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_availabilityzoneprofile_delete_failure(self):
    arglist = ['unknown_availabilityzoneprofile']
    verifylist = [('availabilityzoneprofile', 'unknown_availabilityzoneprofile')]
    self.api_mock.availabilityzoneprofile_list.return_value = {'availability_zone_profiles': []}
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertNotCalled(self.api_mock.availabilityzoneprofile_delete)