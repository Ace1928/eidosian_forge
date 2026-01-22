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
def test_obj_make_obj_compatible_with_relationships(self):
    subobj = MyOwnedObject(baz=1)
    obj = MyObj(rel_object=subobj)
    obj.obj_relationships = {'rel_object': [('1.5', '1.1'), ('1.7', '1.2')]}
    primitive = obj.obj_to_primitive()['versioned_object.data']
    with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
        obj._obj_make_obj_compatible(copy.copy(primitive), '1.8', 'rel_object')
        self.assertFalse(mock_compat.called)
    with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
        obj._obj_make_obj_compatible(copy.copy(primitive), '1.7', 'rel_object')
        mock_compat.assert_called_once_with(primitive['rel_object']['versioned_object.data'], '1.2')
        self.assertEqual('1.2', primitive['rel_object']['versioned_object.version'])
    with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
        obj._obj_make_obj_compatible(copy.copy(primitive), '1.6', 'rel_object')
        mock_compat.assert_called_once_with(primitive['rel_object']['versioned_object.data'], '1.1')
        self.assertEqual('1.1', primitive['rel_object']['versioned_object.version'])
    with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
        obj._obj_make_obj_compatible(copy.copy(primitive), '1.5', 'rel_object')
        mock_compat.assert_called_once_with(primitive['rel_object']['versioned_object.data'], '1.1')
        self.assertEqual('1.1', primitive['rel_object']['versioned_object.version'])
    with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
        _prim = copy.copy(primitive)
        obj._obj_make_obj_compatible(_prim, '1.4', 'rel_object')
        self.assertFalse(mock_compat.called)
        self.assertNotIn('rel_object', _prim)