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
def test_rule_engine_not_any_of_and_direct_mapping(self):
    """Should return user's name and email.

        The CUSTOMER_ASSERTION should successfully have a match in
        MAPPING_LARGE. This will test the case where a requirement
        has `not_any_of`, and direct mapping to a username, no group.

        """
    mapping = mapping_fixtures.MAPPING_LARGE
    assertion = mapping_fixtures.CUSTOMER_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    values = rp.process(assertion)
    self.assertValidMappedUserObject(values)
    user_name = assertion.get('UserName')
    group_ids = values.get('group_ids')
    name = values.get('user', {}).get('name')
    self.assertEqual(user_name, name)
    self.assertEqual([], group_ids)