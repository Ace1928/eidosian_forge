import uuid
from testtools import matchers
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_loaders(self):
    loaders = loading.get_available_plugin_loaders()
    self.assertThat(len(loaders), matchers.GreaterThan(0))
    for loader in loaders.values():
        self.assertIsInstance(loader, loading.BaseLoader)