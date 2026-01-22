from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_obj_list_to_munch(self):
    """Test conversion of a list of objects to a list of dictonaries"""

    class obj0:
        value = 0

    class obj1:
        value = 1
    list = [obj0, obj1]
    new_list = meta.obj_list_to_munch(list)
    self.assertEqual(new_list[0]['value'], 0)
    self.assertEqual(new_list[1]['value'], 1)