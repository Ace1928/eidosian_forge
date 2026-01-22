import argparse
from oslo_utils import netutils
import testtools
from neutronclient.common import exceptions
from neutronclient.common import utils
def test_get_object_item_desired_fields_is_empty(self):

    class Fake(object):

        def __init__(self):
            self.id = 'test_id_1'
            self.name = 'test_name'
            self.test_user = 'test'
    fields = []
    item = Fake()
    actual = utils.get_item_properties(item, fields)
    self.assertEqual((), actual)