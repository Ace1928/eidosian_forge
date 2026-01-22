from testtools import TestCase
from testtools.tags import TagContext
def test_remove_in_child(self):
    parent = TagContext()
    parent.change_tags({'foo'}, set())
    child = TagContext(parent)
    child.change_tags(set(), {'foo'})
    self.assertEqual(set(), child.get_current_tags())
    self.assertEqual({'foo'}, parent.get_current_tags())