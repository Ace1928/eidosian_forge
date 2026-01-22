from ...builtins import cmd_init_shared_repository
from .. import transport_util, ui_testing
def test_init_shared_repository(self):
    cmd = cmd_init_shared_repository()
    cmd.outf = ui_testing.StringIOWithEncoding()
    cmd.run(self.get_url())
    self.assertEqual(1, len(self.connections))