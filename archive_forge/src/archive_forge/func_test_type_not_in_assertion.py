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
def test_type_not_in_assertion(self):
    """Test that if the remote "type" is not in the assertion it fails."""
    mapping = mapping_fixtures.MAPPING_GROUPS_WHITELIST_PASS_THROUGH
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = {uuid.uuid4().hex: uuid.uuid4().hex}
    self.assertRaises(exception.ValidationError, rp.process, assertion)