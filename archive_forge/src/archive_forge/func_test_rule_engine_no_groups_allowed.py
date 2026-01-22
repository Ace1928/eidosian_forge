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
def test_rule_engine_no_groups_allowed(self):
    """Should return user mapped to no groups.

        The EMPLOYEE_ASSERTION should successfully have a match
        in MAPPING_GROUPS_WHITELIST, but 'whitelist' should filter out
        the group values from the assertion and thus map to no groups.

        """
    mapping = mapping_fixtures.MAPPING_GROUPS_WHITELIST
    assertion = mapping_fixtures.EMPLOYEE_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertListEqual(mapped_properties['group_names'], [])
    self.assertListEqual(mapped_properties['group_ids'], [])
    self.assertEqual('tbo', mapped_properties['user']['name'])