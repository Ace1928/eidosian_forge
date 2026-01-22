import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_forbidden_title(self):
    e = exception.Forbidden()
    resp = auth_context.render_exception(e)
    j = jsonutils.loads(resp.body)
    self.assertEqual('Forbidden', e.title)
    self.assertEqual('Forbidden', j['error'].get('title'))