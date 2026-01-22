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
def test_compare_obj_with_comparators(self):
    mock_test = mock.Mock()
    mock_test.assertEqual = mock.Mock()
    comparator = mock.Mock()
    comp_dict = {'foo': comparator}
    my_obj = self.MyComparedObject(foo=1, bar=2)
    my_db_obj = {'foo': 1, 'bar': 2}
    fixture.compare_obj(mock_test, my_obj, my_db_obj, comparators=comp_dict)
    comparator.assert_called_once_with(1, 1)
    mock_test.assertEqual.assert_called_once_with(2, 2)