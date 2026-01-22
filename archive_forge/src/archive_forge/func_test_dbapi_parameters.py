from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_db import options
from oslo_db.tests import base as test_base
def test_dbapi_parameters(self):
    path = self.create_tempfiles([['tmp', b'[database]\nbackend=test_123\n']])[0]
    self.conf(['--config-file', path])
    self.assertEqual('test_123', self.conf.database.backend)