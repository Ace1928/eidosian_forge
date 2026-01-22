import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_properties_for_a_collection_of_objects_max_objects_1(self):
    objects = ['m1', 'm2']
    self._test_get_properties_for_a_collection_of_objects(objects, 1)