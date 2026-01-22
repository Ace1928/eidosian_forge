from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import resource_locks as osc_resource_locks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_lock_delete_multiple(self):
    locks = manila_fakes.FakeResourceLock.create_locks(count=2)
    arglist = [locks[0].id, locks[1].id]
    verifylist = [('lock', [locks[0].id, locks[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.lock.delete.call_count, len(locks))
    self.assertIsNone(result)