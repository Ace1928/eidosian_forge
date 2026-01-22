from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_obj_to_munch_subclass(self):

    class FakeObjDict(dict):
        additional = 1
    obj = FakeObjDict(foo='bar')
    obj_dict = meta.obj_to_munch(obj)
    self.assertIn('additional', obj_dict)
    self.assertIn('foo', obj_dict)
    self.assertEqual(obj_dict['additional'], 1)
    self.assertEqual(obj_dict['foo'], 'bar')