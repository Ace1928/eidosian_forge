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
def test_get_v3_catalog_with_empty_public_url(self):
    service = unit.new_service_ref()
    PROVIDERS.catalog_api.create_service(service['id'], service)
    endpoint = unit.new_endpoint_ref(url='', service_id=service['id'], region_id=None)
    PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint.copy())
    catalog = PROVIDERS.catalog_api.get_v3_catalog(self.user_foo['id'], self.project_bar['id'])
    catalog_endpoint = catalog[0]
    self.assertEqual(service['name'], catalog_endpoint['name'])
    self.assertEqual(service['id'], catalog_endpoint['id'])
    self.assertEqual([], catalog_endpoint['endpoints'])