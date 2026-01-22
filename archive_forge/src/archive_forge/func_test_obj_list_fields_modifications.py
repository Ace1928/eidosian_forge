import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_obj_list_fields_modifications(self):

    @base.VersionedObjectRegistry.register
    class ObjWithList(base.VersionedObject):
        fields = {'list_field': fields.Field(fields.List(fields.Integer()))}
    obj = ObjWithList()

    def set_by_index(val):
        obj.list_field[0] = val

    def append(val):
        obj.list_field.append(val)

    def extend(val):
        obj.list_field.extend([val])

    def add(val):
        obj.list_field = obj.list_field + [val]

    def iadd(val):
        """Test += corner case

            a=a+b and a+=b use different magic methods under the hood:
            first one calls __add__ which clones initial value before the
            assignment, second one call __iadd__ which modifies the initial
            list.
            Assignment should cause coercing in both cases, but __iadd__ may
            corrupt the initial value even if the assignment fails.
            So it should be overridden as well, and this test is needed to
            verify it
            """
        obj.list_field += [val]

    def insert(val):
        obj.list_field.insert(0, val)

    def simple_slice(val):
        obj.list_field[:] = [val]

    def extended_slice(val):
        """Extended slice case

            Extended slice (and regular slices in py3) are handled differently
            thus needing a separate test
            """
        obj.list_field[::2] = [val]
    obj.list_field = ['42']
    set_by_index('1')
    append('2')
    extend('3')
    add('4')
    iadd('5')
    insert('0')
    self.assertEqual([0, 1, 2, 3, 4, 5], obj.list_field)
    simple_slice('10')
    self.assertEqual([10], obj.list_field)
    extended_slice('42')
    self.assertEqual([42], obj.list_field)
    obj.obj_reset_changes()
    self.assertRaises(ValueError, set_by_index, 'abc')
    self.assertRaises(ValueError, append, 'abc')
    self.assertRaises(ValueError, extend, 'abc')
    self.assertRaises(ValueError, add, 'abc')
    self.assertRaises(ValueError, iadd, 'abc')
    self.assertRaises(ValueError, insert, 'abc')
    self.assertRaises(ValueError, simple_slice, 'abc')
    self.assertRaises(ValueError, extended_slice, 'abc')
    self.assertEqual([42], obj.list_field)
    self.assertEqual({}, obj.obj_get_changes())