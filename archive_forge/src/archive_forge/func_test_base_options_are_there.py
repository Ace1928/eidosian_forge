import random
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
def test_base_options_are_there(self):
    options = loading.get_plugin_loader(self.plugin_name).get_options()
    self.assertTrue(set(['client-id', 'client-secret', 'access-token-endpoint', 'access-token-type', 'openid-scope', 'discovery-endpoint']).issubset(set([o.name for o in options])))
    self.assertIn('scope', [o.dest for o in options])