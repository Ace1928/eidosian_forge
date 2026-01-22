from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_dotted_module_no_member(self):
    self.assertEqual(('mod', None, 'sub_mod'), calc_parent_name('mod.sub_mod'))