from io import StringIO
from .. import add, errors, tests
from ..bzr import inventory
def run_action(self, output):
    inv = inventory.Inventory()
    stdout = StringIO()
    action = add.AddAction(to_file=stdout, should_print=bool(output))
    self.apply_redirected(None, stdout, None, action, inv, None, 'path', 'file')
    self.assertEqual(stdout.getvalue(), output)