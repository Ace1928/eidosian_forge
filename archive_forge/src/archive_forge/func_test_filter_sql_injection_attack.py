import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
def test_filter_sql_injection_attack(self):
    """Test against sql injection attack on filters.

        Test Plan:
        - Attempt to get all entities back by passing a two-term attribute
        - Attempt to piggyback filter to damage DB (e.g. drop table)

        """
    users = PROVIDERS.identity_api.list_users()
    self.assertGreater(len(users), 0)
    hints = driver_hints.Hints()
    hints.add_filter('name', "anything' or 'x'='x")
    users = PROVIDERS.identity_api.list_users(hints=hints)
    self.assertEqual(0, len(users))
    group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group = PROVIDERS.identity_api.create_group(group)
    hints = driver_hints.Hints()
    hints.add_filter('name', "x'; drop table group")
    groups = PROVIDERS.identity_api.list_groups(hints=hints)
    self.assertEqual(0, len(groups))
    groups = PROVIDERS.identity_api.list_groups()
    self.assertGreater(len(groups), 0)