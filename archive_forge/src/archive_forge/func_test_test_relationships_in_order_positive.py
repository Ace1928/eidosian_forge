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
def test_test_relationships_in_order_positive(self):
    rels = {'bellsprout': [('1.0', '1.0'), ('1.1', '1.2'), ('1.3', '1.3')]}
    MyObject.obj_relationships = rels
    self.ovc._test_relationships_in_order(MyObject)