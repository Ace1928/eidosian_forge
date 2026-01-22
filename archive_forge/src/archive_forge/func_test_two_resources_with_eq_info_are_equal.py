import testtools
from glanceclient.v1.apiclient import base
def test_two_resources_with_eq_info_are_equal(self):
    r1 = base.Resource(None, {'name': 'joe', 'age': 12})
    r2 = base.Resource(None, {'name': 'joe', 'age': 12})
    self.assertEqual(r1, r2)