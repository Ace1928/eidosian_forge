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
def test_whitelist_pass_through(self):
    mapping = mapping_fixtures.MAPPING_GROUPS_WHITELIST_PASS_THROUGH
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = mapping_fixtures.DEVELOPER_ASSERTION
    mapped_properties = rp.process(assertion)
    self.assertValidMappedUserObject(mapped_properties)
    self.assertEqual('developacct', mapped_properties['user']['name'])
    self.assertEqual('Developer', mapped_properties['group_names'][0]['name'])