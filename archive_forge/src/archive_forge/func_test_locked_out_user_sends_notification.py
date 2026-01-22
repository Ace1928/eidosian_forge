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
def test_locked_out_user_sends_notification(self):
    password = uuid.uuid4().hex
    new_password = uuid.uuid4().hex
    expected_responses = [AssertionError, AssertionError, AssertionError, exception.Unauthorized]
    user_ref = unit.new_user_ref(domain_id=self.domain_id, password=password)
    user_ref = PROVIDERS.identity_api.create_user(user_ref)
    reason_type = exception.AccountLocked.message_format % {'user_id': user_ref['id']}
    expected_reason = {'reasonCode': '401', 'reasonType': reason_type}
    for ex in expected_responses:
        with self.make_request():
            self.assertRaises(ex, PROVIDERS.identity_api.change_password, user_id=user_ref['id'], original_password=new_password, new_password=new_password)
    self._assert_last_audit(None, 'authenticate', None, cadftaxonomy.ACCOUNT_USER, reason=expected_reason)