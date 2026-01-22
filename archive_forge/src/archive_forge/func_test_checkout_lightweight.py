from breezy.builtins import cmd_checkout
from breezy.tests.transport_util import TestCaseWithConnectionHookedTransport
def test_checkout_lightweight(self):
    self.make_branch_and_tree('branch1')
    self.start_logging_connections()
    cmd = cmd_checkout()
    cmd.run(self.get_url('branch1'), 'local', lightweight=True)
    self.assertEqual(1, len(self.connections))