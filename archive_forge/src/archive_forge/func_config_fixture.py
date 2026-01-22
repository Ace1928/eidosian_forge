from oslo_config import fixture
from oslo_utils import uuidutils
from oslotest import base
from oslo_cache import core as cache
@property
def config_fixture(self):
    if not hasattr(self, '_config_fixture'):
        self._config_fixture = self.useFixture(fixture.Config())
        self._config_fixture.config(group='cache', enabled=True)
    return self._config_fixture