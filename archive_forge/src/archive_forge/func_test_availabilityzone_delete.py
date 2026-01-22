import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import availabilityzone
from octaviaclient.osc.v2 import constants
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_availabilityzone_delete(self):
    arglist = [self._availabilityzone.name]
    verifylist = [('availabilityzone', self._availabilityzone.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.availabilityzone_delete.assert_called_with(availabilityzone_name=self._availabilityzone.name)