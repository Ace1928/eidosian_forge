import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_get_object_item_properties_mixed_case_fields(self):

    class Fake(object):

        def __init__(self):
            self.id = 'test_id'
            self.name = 'test_name'
            self.test_user = 'test'
    fields = ('name', 'id', 'test user')
    mixed_fields = ('test user', 'ID')
    item = Fake()
    actual = utils.get_item_properties(item, fields, mixed_fields)
    self.assertEqual(('test_name', 'test_id', 'test'), actual)