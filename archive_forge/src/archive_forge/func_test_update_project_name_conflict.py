import copy
from unittest import mock
import uuid
from oslo_config import cfg
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_update_project_name_conflict(self):
    name = uuid.uuid4().hex
    description = uuid.uuid4().hex
    domain_attrs = {'id': CONF.identity.default_domain_id, 'name': name, 'description': description}
    domain = PROVIDERS.resource_api.create_domain(CONF.identity.default_domain_id, domain_attrs)
    project1 = unit.new_project_ref(domain_id=domain['id'], name=uuid.uuid4().hex)
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    project2 = unit.new_project_ref(domain_id=domain['id'], name=uuid.uuid4().hex)
    project = PROVIDERS.resource_api.create_project(project2['id'], project2)
    self.assertRaises(exception.Conflict, PROVIDERS.resource_api.update_project, project['id'], {'name': project1['name'], 'id': project['id']})