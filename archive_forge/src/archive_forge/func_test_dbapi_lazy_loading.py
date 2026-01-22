from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def test_dbapi_lazy_loading(self):
    dbapi = api.DBAPI('oslo_db.tests.test_api', lazy=True)
    self.assertIsNone(dbapi._backend)
    dbapi.api_class_call1(1, 'abc')
    self.assertIsNotNone(dbapi._backend)