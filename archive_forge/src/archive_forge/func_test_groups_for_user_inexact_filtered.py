import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_groups_for_user_inexact_filtered(self):
    """Test use of filtering doesn't break groups_for_user listing.

        Some backends may use filtering to achieve the list of groups for a
        user, so test that it can combine a second filter.

        Test Plan:

        - Create 10 groups, some with names we can filter on
        - Create 2 users
        - Assign 1 of those users to most of the groups, including some of the
          well known named ones
        - Assign the other user to other groups as spoilers
        - Ensure that when we list groups for users with a filter on the group
          name, both restrictions have been enforced on what is returned.

        """
    group_list, user_list = self._groups_for_user_data()
    hints = driver_hints.Hints()
    hints.add_filter('name', 'Ministry', comparator='contains')
    groups = PROVIDERS.identity_api.list_groups_for_user(user_list[0]['id'], hints=hints)
    self.assertThat(len(groups), matchers.Equals(1))
    self.assertEqual(group_list[6]['id'], groups[0]['id'])
    hints = driver_hints.Hints()
    hints.add_filter('name', 'The', comparator='startswith')
    groups = PROVIDERS.identity_api.list_groups_for_user(user_list[0]['id'], hints=hints)
    self.assertThat(len(groups), matchers.Equals(2))
    self.assertIn(group_list[5]['id'], [groups[0]['id'], groups[1]['id']])
    self.assertIn(group_list[6]['id'], [groups[0]['id'], groups[1]['id']])
    hints.add_filter('name', 'The', comparator='endswith')
    groups = PROVIDERS.identity_api.list_groups_for_user(user_list[0]['id'], hints=hints)
    self.assertThat(len(groups), matchers.Equals(1))
    self.assertEqual(group_list[5]['id'], groups[0]['id'])
    self._delete_test_data('user', user_list)
    self._delete_test_data('group', group_list)