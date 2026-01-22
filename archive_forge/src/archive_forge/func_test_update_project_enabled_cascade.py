import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.resource.backends import sql as resource_sql
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import utils as test_utils
def test_update_project_enabled_cascade(self):
    """Test update_project_cascade.

        Ensures the enabled attribute is correctly updated across
        a simple 3-level projects hierarchy.
        """
    projects_hierarchy = self._create_projects_hierarchy(hierarchy_size=3)
    parent = projects_hierarchy[0]
    parent['enabled'] = False
    parent_ref = PROVIDERS.resource_api.update_project(parent['id'], parent, cascade=True)
    subtree = PROVIDERS.resource_api.list_projects_in_subtree(parent['id'])
    self.assertEqual(2, len(subtree))
    self.assertFalse(parent_ref['enabled'])
    self.assertFalse(subtree[0]['enabled'])
    self.assertFalse(subtree[1]['enabled'])
    parent['enabled'] = True
    parent_ref = PROVIDERS.resource_api.update_project(parent['id'], parent, cascade=True)
    subtree = PROVIDERS.resource_api.list_projects_in_subtree(parent['id'])
    self.assertEqual(2, len(subtree))
    self.assertTrue(parent_ref['enabled'])
    self.assertTrue(subtree[0]['enabled'])
    self.assertTrue(subtree[1]['enabled'])