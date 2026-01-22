import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import flavorprofile
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_flavorprofile_show(self):
    arglist = [self._flavorprofile.id]
    verifylist = [('flavorprofile', self._flavorprofile.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.flavorprofile_show.assert_called_with(flavorprofile_id=self._flavorprofile.id)