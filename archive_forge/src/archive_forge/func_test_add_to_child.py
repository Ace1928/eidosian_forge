from testtools import TestCase
from testtools.tags import TagContext
def test_add_to_child(self):
    parent = TagContext()
    parent.change_tags({'foo'}, set())
    child = TagContext(parent)
    child.change_tags({'bar'}, set())
    self.assertEqual({'foo', 'bar'}, child.get_current_tags())
    self.assertEqual({'foo'}, parent.get_current_tags())