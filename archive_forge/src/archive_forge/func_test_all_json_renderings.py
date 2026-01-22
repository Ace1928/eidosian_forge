import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_all_json_renderings(self):
    """Everything callable in the exception module should be renderable.

        ... except for the base error class (exception.Error), which is not
        user-facing.

        This test provides a custom message to bypass docstring parsing, which
        should be tested separately.

        """
    for cls in [x for x in exception.__dict__.values() if callable(x)]:
        if cls is not exception.Error and isinstance(cls, exception.Error):
            self.assertValidJsonRendering(cls(message='Overridden.'))