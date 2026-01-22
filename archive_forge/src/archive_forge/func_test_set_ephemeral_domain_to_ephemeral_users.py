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
def test_set_ephemeral_domain_to_ephemeral_users(self):
    """Test auto assigning service domain to ephemeral users.

        Test that ephemeral users will always become members of federated
        service domain. The check depends on ``type`` value which must be set
        to ``ephemeral`` in case of ephemeral user.

        """
    mapping = mapping_fixtures.MAPPING_EPHEMERAL_USER_LOCAL_DOMAIN
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = mapping_fixtures.CONTRACTOR_ASSERTION
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertValidMappedUserObject(mapped_properties)