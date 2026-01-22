import flask
import uuid
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.auth.plugins import mapped
import keystone.conf
from keystone import exception
from keystone.federation import utils as mapping_utils
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from unittest import mock
def test_rule_engine_for_groups_and_domain(self):
    """Should return user's groups and group domain.

        The GROUP_DOMAIN_ASSERTION should successfully have a match in
        MAPPING_GROUPS_DOMAIN_OF_USER. This will test the case where a groups
        with its domain will exist`, and return user's groups and group domain.

        """
    mapping = mapping_fixtures.MAPPING_GROUPS_DOMAIN_OF_USER
    assertion = mapping_fixtures.GROUPS_DOMAIN_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    values = rp.process(assertion)
    self.assertValidMappedUserObject(values)
    user_name = assertion.get('openstack_user')
    user_groups = ['group1', 'group2']
    groups = values.get('group_names', {})
    group_list = [g.get('name') for g in groups]
    group_ids = values.get('group_ids')
    name = values.get('user', {}).get('name')
    self.assertEqual(user_name, name)
    self.assertEqual(user_groups, group_list)
    self.assertEqual([], group_ids)