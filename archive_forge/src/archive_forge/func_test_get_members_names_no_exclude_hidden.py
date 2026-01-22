import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_get_members_names_no_exclude_hidden(self):
    obj = TestObject()
    members = list(reflection.get_member_names(obj, exclude_hidden=False))
    members = [member for member in members if not member.startswith('__')]
    self.assertEqual(['_hello', 'hi'], sorted(members))