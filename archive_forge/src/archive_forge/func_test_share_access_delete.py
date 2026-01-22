from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data(True, False)
def test_share_access_delete(self, unrestrict):
    arglist = [self.share.id, self.access_rule.id]
    verifylist = [('share', self.share.id), ('id', self.access_rule.id)]
    deny_kwargs = {}
    if unrestrict:
        arglist.append('--unrestrict')
        verifylist.append(('unrestrict', unrestrict))
        deny_kwargs['unrestrict'] = unrestrict
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.shares_mock.get.assert_called_with(self.share.id)
    self.share.deny.assert_called_with(self.access_rule.id, **deny_kwargs)
    self.assertIsNone(result)