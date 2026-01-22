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
def test_update_group_domain_id(self):
    """Call ``PATCH /groups/{group_id}`` with domain_id.

        A group's `domain_id` is immutable. Ensure that any attempts to update
        the `domain_id` of a group fails.
        """
    self.group['domain_id'] = CONF.identity.default_domain_id
    self.patch('/groups/%(group_id)s' % {'group_id': self.group['id']}, body={'group': self.group}, expected_status=exception.ValidationError.code)