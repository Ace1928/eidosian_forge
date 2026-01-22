import copy
import uuid
from keystone.application_credential import schema as app_cred_schema
from keystone.assignment import schema as assignment_schema
from keystone.catalog import schema as catalog_schema
from keystone.common import validation
from keystone.common.validation import parameter_types
from keystone.common.validation import validators
from keystone.credential import schema as credential_schema
from keystone import exception
from keystone.federation import schema as federation_schema
from keystone.identity.backends import resource_options as ro
from keystone.identity import schema as identity_schema
from keystone.limit import schema as limit_schema
from keystone.oauth1 import schema as oauth1_schema
from keystone.policy import schema as policy_schema
from keystone.resource import schema as resource_schema
from keystone.tests import unit
from keystone.trust import schema as trust_schema
def test_validate_registered_limit_update_request_with_invalid_input(self):
    _INVALID_FORMATS = [{'service_id': 'fake_id'}, {'region_id': 123}, {'resource_name': 123}, {'resource_name': ''}, {'resource_name': 'a' * 256}, {'default_limit': 'not_int'}, {'default_limit': -10}, {'default_limit': 10000000000000000}, {'description': 123}]
    for invalid_desc in _INVALID_FORMATS:
        request_to_validate = {'service_id': uuid.uuid4().hex, 'region_id': 'RegionOne', 'resource_name': 'volume', 'default_limit': 10, 'description': 'test description'}
        request_to_validate.update(invalid_desc)
        self.assertRaises(exception.SchemaValidationError, self.update_registered_limits_validator.validate, request_to_validate)