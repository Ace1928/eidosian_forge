import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import flavorprofile
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_flavorprofile_set(self):
    arglist = [self._flavorprofile.id, '--name', 'new_name']
    verifylist = [('flavorprofile', self._flavorprofile.id), ('name', 'new_name')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.flavorprofile_set.assert_called_with(self._flavorprofile.id, json={'flavorprofile': {'name': 'new_name'}})