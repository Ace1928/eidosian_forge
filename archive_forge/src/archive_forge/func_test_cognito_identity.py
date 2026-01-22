from boto.cognito.identity.exceptions import ResourceNotFoundException
from tests.integration.cognito import CognitoTest
def test_cognito_identity(self):
    response = self.cognito_identity.list_identity_pools(max_results=5)
    expected_identity = {'IdentityPoolId': self.identity_pool_id, 'IdentityPoolName': self.identity_pool_name}
    self.assertIn(expected_identity, response['IdentityPools'])
    response = self.cognito_identity.describe_identity_pool(identity_pool_id=self.identity_pool_id)
    self.assertEqual(response['IdentityPoolName'], self.identity_pool_name)
    self.assertEqual(response['IdentityPoolId'], self.identity_pool_id)
    self.assertFalse(response['AllowUnauthenticatedIdentities'])