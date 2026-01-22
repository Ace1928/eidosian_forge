import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_create_duplicate_user_name_in_different_domains(self):
    new_domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
    user1 = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user2 = unit.new_user_ref(name=user1['name'], domain_id=new_domain['id'])
    PROVIDERS.identity_api.create_user(user1)
    PROVIDERS.identity_api.create_user(user2)