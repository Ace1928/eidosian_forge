import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_invalid_password_sends_notification(self):
    password = uuid.uuid4().hex
    invalid_password = '1'
    regex = CONF.security_compliance.password_regex_description
    reason_type = exception.PasswordRequirementsValidationError.message_format % {'detail': regex}
    expected_reason = {'reasonCode': '400', 'reasonType': reason_type}
    user_ref = unit.new_user_ref(domain_id=self.domain_id, password=password)
    user_ref = PROVIDERS.identity_api.create_user(user_ref)
    with self.make_request():
        self.assertRaises(exception.PasswordValidationError, PROVIDERS.identity_api.change_password, user_id=user_ref['id'], original_password=password, new_password=invalid_password)
    self._assert_last_audit(user_ref['id'], UPDATED_OPERATION, 'user', cadftaxonomy.SECURITY_ACCOUNT_USER, reason=expected_reason)