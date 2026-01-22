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
def test_get_dependencies(self):
    self._add_class(self.obj_classes, MyExtraObject)
    MyObject.fields['subob'] = fields.ObjectField('MyExtraObject')
    MyExtraObject.VERSION = '1.0'
    tree = {}
    self.ovc._get_dependencies(tree, MyObject)
    expected_tree = {'MyObject': {'MyExtraObject': '1.0'}}
    self.assertEqual(expected_tree, tree, '_get_dependencies() did not generate a correct dependency tree.')