import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def test_delete_with_multiple_resources(self):

    def _mock_fwaas(*args, **kwargs):
        return {'id': args[0]}
    self.networkclient.find_firewall_group.side_effect = _mock_fwaas
    self.networkclient.find_firewall_policy.side_effect = _mock_fwaas
    self.networkclient.find_firewall_rule.side_effect = _mock_fwaas
    target1 = 'target1'
    target2 = 'target2'
    arglist = [target1, target2]
    verifylist = [(self.res, [target1, target2])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)
    self.assertEqual(2, self.mocked.call_count)
    for idx, reference in enumerate([target1, target2]):
        actual = ''.join(self.mocked.call_args_list[idx][0][0])
        self.assertEqual(reference, actual)