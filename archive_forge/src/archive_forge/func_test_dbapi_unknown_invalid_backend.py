from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def test_dbapi_unknown_invalid_backend(self):
    self.assertRaises(ImportError, api.DBAPI, 'tests.unit.db.not_existent')