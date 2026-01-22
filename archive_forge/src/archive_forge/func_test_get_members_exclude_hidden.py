import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_get_members_exclude_hidden(self):
    obj = TestObject()
    members = list(reflection.get_members(obj, exclude_hidden=True))
    self.assertEqual(1, len(members))