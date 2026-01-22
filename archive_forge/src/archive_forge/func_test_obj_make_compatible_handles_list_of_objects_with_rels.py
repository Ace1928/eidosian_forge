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
def test_obj_make_compatible_handles_list_of_objects_with_rels(self):
    subobj = MyOwnedObject(baz=1)
    obj = MyObj(rel_objects=[subobj])
    obj.obj_relationships = {'rel_objects': [('1.0', '1.123')]}

    def fake_make_compat(primitive, version, **k):
        self.assertEqual('1.123', version)
        self.assertIn('baz', primitive)
    with mock.patch.object(subobj, 'obj_make_compatible') as mock_mc:
        mock_mc.side_effect = fake_make_compat
        obj.obj_to_primitive('1.0')
        self.assertTrue(mock_mc.called)