import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_string_to_sign(self):
    with mock.patch('hashlib.sha256') as mock_sha256:
        mock_sha256.return_value.hexdigest.return_value = 'chksum_of_canonical_request'
        to_sign = self.signer._get_string_to_sign({}, {}, self.now, method='GET', path='/', data=None)
    self.assertEqual(to_sign, 'AWS4-HMAC-SHA256\n20150304T173452Z\n20150304/my_region/my_service/aws4_request\nchksum_of_canonical_request')