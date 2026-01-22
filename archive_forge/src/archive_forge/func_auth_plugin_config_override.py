import http.client
from oslo_serialization import jsonutils
import webtest
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def auth_plugin_config_override(self, methods=None, **method_classes):
    self.useFixture(ksfixtures.ConfigAuthPlugins(self.config_fixture, methods, **method_classes))