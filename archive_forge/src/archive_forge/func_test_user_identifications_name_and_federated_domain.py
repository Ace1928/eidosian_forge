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
def test_user_identifications_name_and_federated_domain(self):
    """Test various mapping options and how users are identified.

        This test calls mapped.setup_username() for propagating user object.

        Test plan:
        - Check if the user has proper domain ('federated') set
        - Check if the user has propert type set ('ephemeral')
        - Check if user's name is properly mapped from the assertion
        - Check if the unique_id and display_name are properly set

        """
    mapping = mapping_fixtures.MAPPING_USER_IDS
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = mapping_fixtures.EMPLOYEE_ASSERTION
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertValidMappedUserObject(mapped_properties)
    resource_api_mock = mock.patch('keystone.resource.core.DomainConfigManager')
    idp_domain_id = uuid.uuid4().hex
    user_domain_id = mapped_properties['user']['domain']['id']
    mapped.validate_and_prepare_federated_user(mapped_properties, idp_domain_id, resource_api_mock)
    self.assertEqual('tbo', mapped_properties['user']['name'])
    self.assertEqual('abc123%40example.com', mapped_properties['user']['id'])
    self.assertEqual(user_domain_id, mapped_properties['user']['domain']['id'])