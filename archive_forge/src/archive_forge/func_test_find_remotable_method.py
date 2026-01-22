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
def test_find_remotable_method(self):
    method = self.ovc._find_remotable_method(MyObject, MyObject.remotable_method)
    self.assertEqual(MyObject.remotable_method.original_fn, method, '_find_remotable_method() did not find the remotable method of MyObject.')