from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_two_resources_with_diff_type_are_not_equal(self):
    r1 = base.Resource(None, {'id': 1})
    r2 = HumanResource(None, {'id': 1})
    self.assertNotEqual(r1, r2)