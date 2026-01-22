import http.client
import uuid
from keystone.common import provider_api
from keystone.common.validation import validators
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_default_limit_model_response_schema(self):
    schema = {'type': 'object', 'properties': {'model': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'description': {'type': 'string'}}, 'required': ['name', 'description'], 'additionalProperties': False}}, 'required': ['model'], 'additionalProperties': False}
    validator = validators.SchemaValidator(schema)
    response = self.get('/limits/model')
    validator.validate(response.json_body)