from unittest import mock
from neutron_lib.objects import utils as obj_utils
from neutron_lib.tests import _base as base
def test_get_objects_with_filters_not_in(self):

    class FakeColumn(object):

        def __init__(self, column):
            self.column = column

        def in_(self, value):
            self.value = value
            return self

        def __invert__(self):
            return list(set(self.column) - set(self.value))
    filter_obj = obj_utils.NotIn([1, 2, 3])
    fake_column = FakeColumn([1, 2, 4, 5])
    self.assertEqual([4, 5], sorted(filter_obj.filter(fake_column)))
    fake_column = FakeColumn([1, 2])
    self.assertEqual([], filter_obj.filter(fake_column))
    fake_column = FakeColumn([4, 5])
    self.assertEqual([4, 5], sorted(filter_obj.filter(fake_column)))