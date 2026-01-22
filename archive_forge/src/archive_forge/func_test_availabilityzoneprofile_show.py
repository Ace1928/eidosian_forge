import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzoneprofile
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_availabilityzoneprofile_show(self):
    arglist = [self._availabilityzoneprofile.id]
    verifylist = [('availabilityzoneprofile', self._availabilityzoneprofile.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.availabilityzoneprofile_show.assert_called_with(availabilityzoneprofile_id=self._availabilityzoneprofile.id)