import http.client as http
from oslo_serialization import jsonutils
import webob
from glance.common import auth
from glance.common import exception
from glance.tests import utils
def test_get_plugin_from_strategy_keystone(self):
    strategy = auth.get_plugin_from_strategy('keystone')
    self.assertIsInstance(strategy, auth.KeystoneStrategy)
    self.assertTrue(strategy.configure_via_auth)