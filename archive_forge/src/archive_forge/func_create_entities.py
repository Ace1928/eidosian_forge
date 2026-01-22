from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def create_entities(self, entity_pattern):
    """Create the entities specified in the test plan.

        Process the 'entities' key in the test plan, creating the requested
        entities. Each created entity will be added to the array of entities
        stored in the returned test_data object, e.g.:

        test_data['users'] = [user[0], user[1]....]

        """
    test_data = {}
    for entity in ['users', 'groups', 'domains', 'projects', 'roles']:
        test_data[entity] = []
    if 'domains' in entity_pattern:
        self._handle_domain_spec(test_data, entity_pattern['domains'])
    if 'roles' in entity_pattern:
        for _ in range(entity_pattern['roles']):
            test_data['roles'].append(self._create_role())
    return test_data