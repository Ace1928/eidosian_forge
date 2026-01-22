from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def execute_assignment_plan(self, test_plan):
    """Create entities, assignments and execute the test plan.

        The standard method to call to create entities and assignments and
        execute the tests as specified in the test_plan. The test_data
        dict is returned so that, if required, the caller can execute
        additional manual tests with the entities and assignments created.

        """
    test_data = self.create_entities(test_plan['entities'])
    if 'implied_roles' in test_plan:
        self.create_implied_roles(test_plan['implied_roles'], test_data)
    if 'group_memberships' in test_plan:
        self.create_group_memberships(test_plan['group_memberships'], test_data)
    if 'assignments' in test_plan:
        test_data = self.create_assignments(test_plan['assignments'], test_data)
    self.execute_assignment_cases(test_plan, test_data)
    return test_data