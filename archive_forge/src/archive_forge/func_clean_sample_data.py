import uuid
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
from keystone.identity.mapping_backends import mapping
from keystone.tests import unit
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit import test_backend_sql
def clean_sample_data(self):
    if hasattr(self, 'domainA'):
        self.domainA['enabled'] = False
        PROVIDERS.resource_api.update_domain(self.domainA['id'], self.domainA)
        PROVIDERS.resource_api.delete_domain(self.domainA['id'])
    if hasattr(self, 'domainB'):
        self.domainB['enabled'] = False
        PROVIDERS.resource_api.update_domain(self.domainB['id'], self.domainB)
        PROVIDERS.resource_api.delete_domain(self.domainB['id'])