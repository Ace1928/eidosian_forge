import http.client
from oslo_serialization import jsonutils
import webtest
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def public_request(self, **kwargs):
    return self._request(app=self.public_app, **kwargs)