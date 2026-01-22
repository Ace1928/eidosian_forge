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
def test_list_users_call_count(self):
    """There should not be O(N) queries."""
    for i in range(10):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.identity_api.create_user(user)

    class CallCounter(object):

        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls = 0

        def query_counter(self, query):
            self.calls += 1
    counter = CallCounter()
    sqlalchemy.event.listen(sqlalchemy.orm.query.Query, 'before_compile', counter.query_counter)
    first_call_users = PROVIDERS.identity_api.list_users()
    first_call_counter = counter.calls
    for i in range(10):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.identity_api.create_user(user)
    counter.reset()
    second_call_users = PROVIDERS.identity_api.list_users()
    self.assertNotEqual(len(first_call_users), len(second_call_users))
    self.assertEqual(first_call_counter, counter.calls)
    self.assertEqual(3, counter.calls)