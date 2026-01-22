from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
def test_update_saml_provider(self):
    self.set_http_response(status_code=200)
    self.service_connection.update_saml_provider('arn', 'doc')
    self.assert_request_parameters({'Action': 'UpdateSAMLProvider', 'SAMLMetadataDocument': 'doc', 'SAMLProviderArn': 'arn'}, ignore_params_values=['Version'])