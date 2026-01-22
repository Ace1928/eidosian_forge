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
def test_user_identifications_name(self):
    """Test various mapping options and how users are identified.

        This test calls mapped.setup_username() for propagating user object.

        Test plan:
        - Check if the user has proper domain ('federated') set
        - Check if the user has property type set ('ephemeral')
        - Check if user's name is properly mapped from the assertion
        - Check if unique_id is properly set and equal to display_name,
        as it was not explicitly specified in the mapping.

        """
    mapping = mapping_fixtures.MAPPING_USER_IDS
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = mapping_fixtures.CONTRACTOR_ASSERTION
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertValidMappedUserObject(mapped_properties)
    self.assertEqual('jsmith', mapped_properties['user']['name'])
    resource_api_mock = mock.patch('keystone.resource.core.DomainConfigManager')
    idp_domain_id = uuid.uuid4().hex
    mapped.validate_and_prepare_federated_user(mapped_properties, idp_domain_id, resource_api_mock)
    self.assertEqual('jsmith', mapped_properties['user']['id'])
    self.assertEqual('jsmith', mapped_properties['user']['name'])
    self.assertEqual(idp_domain_id, mapped_properties['user']['domain']['id'])