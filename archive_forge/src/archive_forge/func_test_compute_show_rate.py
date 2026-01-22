from openstackclient.common import limits
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
def test_compute_show_rate(self):
    arglist = ['--rate']
    verifylist = [('is_rate', True)]
    cmd = limits.ShowLimits(self.app, None)
    parsed_args = self.check_parser(cmd, arglist, verifylist)
    columns, data = cmd.take_action(parsed_args)
    ret_limits = list(data)
    compute_reference_limits = self.fake_limits.rate_limits()
    self.assertEqual(self.rate_columns, columns)
    self.assertEqual(compute_reference_limits, ret_limits)
    self.assertEqual(3, len(ret_limits))