import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_v4_signature_contains_credential_scope(self):
    with mock.patch('libcloud.common.aws.AWSRequestSignerAlgorithmV4._get_credential_scope') as mock_get_creds:
        mock_get_creds.return_value = 'my_credential_scope'
        sig = self.signer._get_authorization_v4_header(params={}, headers={}, dt=self.now)
    self.assertIn('Credential=my_key/my_credential_scope, ', sig)