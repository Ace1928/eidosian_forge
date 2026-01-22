import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_validation_error(self):
    target = uuid.uuid4().hex
    attribute = uuid.uuid4().hex
    e = exception.ValidationError(target=target, attribute=attribute)
    self.assertValidJsonRendering(e)
    self.assertIn(target, str(e))
    self.assertIn(attribute, str(e))