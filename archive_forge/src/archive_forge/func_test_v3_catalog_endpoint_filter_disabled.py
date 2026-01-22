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
def test_v3_catalog_endpoint_filter_disabled(self):
    self.config_fixture.config(group='endpoint_filter', return_all_endpoints_if_no_filter=True)
    srv_1 = unit.new_service_ref()
    PROVIDERS.catalog_api.create_service(srv_1['id'], srv_1)
    endpoint_1 = unit.new_endpoint_ref(service_id=srv_1['id'], region_id=None)
    PROVIDERS.catalog_api.create_endpoint(endpoint_1['id'], endpoint_1)
    srv_2 = unit.new_service_ref()
    PROVIDERS.catalog_api.create_service(srv_2['id'], srv_2)
    catalog_ref = PROVIDERS.catalog_api.get_v3_catalog(uuid.uuid4().hex, self.project_bar['id'])
    self.assertThat(catalog_ref, matchers.HasLength(2))
    srv_id_list = [catalog_ref[0]['id'], catalog_ref[1]['id']]
    self.assertCountEqual([srv_1['id'], srv_2['id']], srv_id_list)