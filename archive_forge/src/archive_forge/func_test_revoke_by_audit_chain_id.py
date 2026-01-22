import datetime
from unittest import mock
import uuid
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import revoke_model
from keystone.revoke.backends import sql
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_backend_sql
from keystone.token import provider
def test_revoke_by_audit_chain_id(self):
    revocation_backend = sql.Revoke()
    audit_id = provider.random_urlsafe_str()
    token = _sample_blank_token()
    token['audit_id'] = audit_id
    token['audit_chain_id'] = audit_id
    self._assertTokenNotRevoked(token)
    self.assertEqual(0, len(revocation_backend.list_events(token=token)))
    PROVIDERS.revoke_api.revoke_by_audit_chain_id(audit_id)
    self._assertTokenRevoked(token)
    self.assertEqual(1, len(revocation_backend.list_events(token=token)))