import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import l7policy
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_l7policy_set(self):
    arglist = [self._l7po.id, '--name', 'new_name']
    verifylist = [('l7policy', self._l7po.id), ('name', 'new_name')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.l7policy_set.assert_called_with(self._l7po.id, json={'l7policy': {'name': 'new_name'}})