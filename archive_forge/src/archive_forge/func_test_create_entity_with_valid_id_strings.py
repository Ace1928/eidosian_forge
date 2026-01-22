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
def test_create_entity_with_valid_id_strings(self):
    """Validate acceptable id strings."""
    valid_id_strings = [str(uuid.uuid4()), uuid.uuid4().hex, 'default']
    for valid_id in valid_id_strings:
        request_to_validate = {'name': self.resource_name, 'id_string': valid_id}
        self.create_schema_validator.validate(request_to_validate)