from boto.cognito.sync.exceptions import ResourceNotFoundException
from tests.integration.cognito import CognitoTest
def test_cognito_sync(self):
    response = self.cognito_sync.describe_identity_pool_usage(identity_pool_id=self.identity_pool_id)
    identity_pool_usage = response['IdentityPoolUsage']
    self.assertEqual(identity_pool_usage['SyncSessionsCount'], None)
    self.assertEqual(identity_pool_usage['DataStorage'], 0)