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
def test_get_projects_in_subtree_as_ids_with_large_tree(self):
    """Check project hierarchy is returned correctly in large tree.

        With a large hierarchy we need to enforce the projects
        are returned in the correct order (illustrated below).

        Tree we will create::

               +------p1------+
               |              |
            +---p3---+      +-p2-+
            |        |      |    |
            p7    +-p6-+   p5    p4
            |     |    |
            p10   p9   p8
                  |
                 p11
        """
    p1, p2, p4 = self._create_projects_hierarchy(hierarchy_size=3)
    p5 = self._create_projects_hierarchy(hierarchy_size=1, parent_project_id=p2['id'])[0]
    p3, p6, p8 = self._create_projects_hierarchy(hierarchy_size=3, parent_project_id=p1['id'])
    p9, p11 = self._create_projects_hierarchy(hierarchy_size=2, parent_project_id=p6['id'])
    p7, p10 = self._create_projects_hierarchy(hierarchy_size=2, parent_project_id=p3['id'])
    expected_projects = {p2['id']: {p5['id']: None, p4['id']: None}, p3['id']: {p7['id']: {p10['id']: None}, p6['id']: {p9['id']: {p11['id']: None}, p8['id']: None}}}
    prjs_hierarchy = PROVIDERS.resource_api.get_projects_in_subtree_as_ids(p1['id'])
    self.assertDictEqual(expected_projects, prjs_hierarchy)