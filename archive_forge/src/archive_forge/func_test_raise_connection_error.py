from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def test_raise_connection_error(self):
    self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__})
    self.test_db_api.error_counter = 5
    self.assertRaises(exception.DBConnectionError, self.dbapi._api_raise)