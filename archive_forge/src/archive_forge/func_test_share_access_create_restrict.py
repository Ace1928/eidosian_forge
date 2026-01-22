from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data({'lock_visibility': True, 'lock_deletion': True, 'lock_reason': 'testing resource locks'}, {'lock_visibility': False, 'lock_deletion': True, 'lock_reason': None}, {'lock_visibility': True, 'lock_deletion': False, 'lock_reason': None})
@ddt.unpack
def test_share_access_create_restrict(self, lock_visibility, lock_deletion, lock_reason):
    arglist = [self.share.id, 'user', 'demo', '--properties', 'key=value']
    verifylist = [('share', self.share.id), ('access_type', 'user'), ('access_to', 'demo'), ('properties', ['key=value'])]
    allow_call_kwargs = {}
    if lock_visibility:
        arglist.append('--lock-visibility')
        verifylist.append(('lock_visibility', lock_visibility))
        allow_call_kwargs['lock_visibility'] = lock_visibility
    if lock_deletion:
        arglist.append('--lock-deletion')
        verifylist.append(('lock_deletion', lock_deletion))
        allow_call_kwargs['lock_deletion'] = lock_deletion
    if lock_reason:
        arglist.append('--lock-reason')
        arglist.append(lock_reason)
        verifylist.append(('lock_reason', lock_reason))
        allow_call_kwargs['lock_reason'] = lock_reason
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.shares_mock.get.assert_called_with(self.share.id)
    self.share.allow.assert_called_with(access_type='user', access='demo', access_level=None, metadata={'key': 'value'}, **allow_call_kwargs)
    self.assertEqual(ACCESS_RULE_ATTRIBUTES, columns)
    self.assertCountEqual(self.access_rule._info.values(), data)