from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
import stevedore
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.loading._plugins.identity import v2
from keystoneauth1.loading._plugins.identity import v3
from keystoneauth1.tests.unit.loading import utils
def test_common_conf_options(self):
    opts = loading.get_auth_common_conf_options()
    self.assertEqual(2, len(opts))
    auth_type = [o for o in opts if o.name == 'auth_type'][0]
    self.assertEqual(1, len(auth_type.deprecated_opts))
    self.assertIsInstance(auth_type.deprecated_opts[0], cfg.DeprecatedOpt)