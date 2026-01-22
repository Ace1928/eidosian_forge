import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_update_user_domain_id(self):
    """Call ``PATCH /users/{user_id}`` with domain_id.

        A user's `domain_id` is immutable. Ensure that any attempts to update
        the `domain_id` of a user fails.
        """
    user = unit.new_user_ref(domain_id=self.domain['id'])
    user = PROVIDERS.identity_api.create_user(user)
    user['domain_id'] = CONF.identity.default_domain_id
    self.patch('/users/%(user_id)s' % {'user_id': user['id']}, body={'user': user}, expected_status=exception.ValidationError.code)