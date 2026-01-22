from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_undotted_module_no_member(self):
    err = self.assertRaises(AssertionError, calc_parent_name, 'mod_name')
    self.assertEqual("No parent object for top-level module 'mod_name'", err.args[0])