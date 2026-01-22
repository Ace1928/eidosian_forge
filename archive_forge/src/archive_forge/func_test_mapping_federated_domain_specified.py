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
def test_mapping_federated_domain_specified(self):
    """Test mapping engine when domain 'ephemeral' is explicitly set.

        For that, we use mapping rule MAPPING_EPHEMERAL_USER and assertion
        EMPLOYEE_ASSERTION

        """
    mapping = mapping_fixtures.MAPPING_EPHEMERAL_USER
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = mapping_fixtures.EMPLOYEE_ASSERTION
    mapped_properties = rp.process(assertion)
    self.assertIsNotNone(mapped_properties)
    self.assertValidMappedUserObject(mapped_properties)