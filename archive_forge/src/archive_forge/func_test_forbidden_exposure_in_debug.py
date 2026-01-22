import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_forbidden_exposure_in_debug(self):
    self.config_fixture.config(debug=True, insecure_debug=True)
    risky_info = uuid.uuid4().hex
    e = exception.Forbidden(message=risky_info)
    self.assertValidJsonRendering(e)
    self.assertIn(risky_info, str(e))