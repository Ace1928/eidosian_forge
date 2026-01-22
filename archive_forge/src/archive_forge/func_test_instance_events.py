import unittest
from traits.api import Any, HasStrictTraits, Str
def test_instance_events(self):
    foo = self.foo
    foo.add_trait('val2', Str(event='the_trait'))
    foo.val2 = 'CHANGE2'
    values = ['CHANGE2']
    self.assertEqual(self.change_events, values)