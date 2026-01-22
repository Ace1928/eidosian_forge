from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_two_resources_with_same_id_and_info_are_equal(self):
    r1 = base.Resource(None, {'id': 1, 'name': 'hello'})
    r2 = base.Resource(None, {'id': 1, 'name': 'hello'})
    self.assertEqual(r1, r2)