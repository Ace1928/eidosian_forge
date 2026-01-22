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
def test_compare_obj_with_dt(self):
    mock_test = mock.Mock()
    mock_test.assertEqual = mock.Mock()
    dt = datetime.datetime(1955, 11, 5, tzinfo=iso8601.iso8601.UTC)
    replaced_dt = dt.replace(tzinfo=None)
    my_obj = self.MyComparedObjectWithTZ(tzfield=dt)
    my_db_obj = {'tzfield': replaced_dt}
    fixture.compare_obj(mock_test, my_obj, my_db_obj)
    mock_test.assertEqual.assert_called_once_with(replaced_dt, replaced_dt)