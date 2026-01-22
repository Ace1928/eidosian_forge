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
def test_rule_engine_regex_whitelist(self):
    mapping = mapping_fixtures.MAPPING_GROUPS_WHITELIST_REGEX
    assertion = mapping_fixtures.EMPLOYEE_PARTTIME_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    mapped = rp.process(assertion)
    expected = {'user': {'type': 'ephemeral'}, 'projects': [], 'group_ids': [], 'group_names': [{'name': 'Employee', 'domain': {'id': mapping_fixtures.FEDERATED_DOMAIN}}, {'name': 'PartTimeEmployee', 'domain': {'id': mapping_fixtures.FEDERATED_DOMAIN}}]}
    self.assertEqual(expected, mapped)