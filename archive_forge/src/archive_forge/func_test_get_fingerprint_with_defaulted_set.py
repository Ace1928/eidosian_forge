import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_get_fingerprint_with_defaulted_set(self):

    class ClassWithDefaultedSetField(base.VersionedObject):
        VERSION = 1.0
        fields = {'empty_default': fields.SetOfIntegersField(default=set()), 'non_empty_default': fields.SetOfIntegersField(default={1, 2})}
    self._add_class(self.obj_classes, ClassWithDefaultedSetField)
    expected = '1.0-bcc44920f2f727eca463c6eb4fe8445b'
    actual = self.ovc._get_fingerprint(ClassWithDefaultedSetField.__name__)
    self.assertEqual(expected, actual)