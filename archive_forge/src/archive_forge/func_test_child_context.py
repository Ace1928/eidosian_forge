from testtools import TestCase
from testtools.tags import TagContext
def test_child_context(self):
    parent = TagContext()
    parent.change_tags({'foo'}, set())
    child = TagContext(parent)
    self.assertEqual(parent.get_current_tags(), child.get_current_tags())