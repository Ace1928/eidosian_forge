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
def test_create_trust(self):
    trustor = unit.new_user_ref(domain_id=self.domain_id)
    trustor = PROVIDERS.identity_api.create_user(trustor)
    trustee = unit.new_user_ref(domain_id=self.domain_id)
    trustee = PROVIDERS.identity_api.create_user(trustee)
    role_ref = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
    trust_ref = unit.new_trust_ref(trustor['id'], trustee['id'])
    PROVIDERS.trust_api.create_trust(trust_ref['id'], trust_ref, [role_ref])
    self._assert_last_note(trust_ref['id'], CREATED_OPERATION, 'OS-TRUST:trust')
    self._assert_last_audit(trust_ref['id'], CREATED_OPERATION, 'OS-TRUST:trust', cadftaxonomy.SECURITY_TRUST)