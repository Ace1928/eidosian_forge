import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_unexpected_error_no_debug(self):
    self.config_fixture.config(debug=False)
    e = exception.UnexpectedError(exception=self.exc_str)
    self.assertNotIn(self.exc_str, str(e))