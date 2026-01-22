from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_dotted_member(self):
    self.assertEqual(('mod_name', 'attr1', 'attr2'), calc_parent_name('mod_name', 'attr1.attr2'))