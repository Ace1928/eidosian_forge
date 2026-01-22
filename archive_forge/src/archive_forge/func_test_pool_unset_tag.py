import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import pool as pool
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
def test_pool_unset_tag(self):
    self.api_mock.pool_set.reset_mock()
    self.api_mock.pool_show.return_value = {'tags': ['foo', 'bar']}
    arglist = [self._po.id, '--tag', 'foo']
    verifylist = [('pool', self._po.id), ('tags', ['foo'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.pool_set.assert_called_once_with(self._po.id, json={'pool': {'tags': ['bar']}})