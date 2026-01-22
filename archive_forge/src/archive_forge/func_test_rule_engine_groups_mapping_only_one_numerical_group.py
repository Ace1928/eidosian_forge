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
def test_rule_engine_groups_mapping_only_one_numerical_group(self):
    """Test mapping engine when groups is explicitly set.

        If the groups list has only one group,
        test if the transformation is done correctly

        """
    mapping = mapping_fixtures.MAPPING_GROUPS_WITH_EMAIL
    assertion = mapping_fixtures.GROUPS_ASSERTION_ONLY_ONE_NUMERICAL_GROUP
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertEqual('jsmith', mapped_properties['user']['name'])
    self.assertEqual('jill@example.com', mapped_properties['user']['email'])
    self.assertEqual('1234', mapped_properties['group_names'][0]['name'])