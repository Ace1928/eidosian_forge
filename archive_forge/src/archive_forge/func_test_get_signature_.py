import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_signature_(self):

    def _sign(key, msg, hex=False):
        if hex:
            return 'H|{}|{}'.format(key, msg)
        else:
            return '{}|{}'.format(key, msg)
    with mock.patch('libcloud.common.aws.AWSRequestSignerAlgorithmV4._get_key_to_sign_with') as mock_get_key:
        with mock.patch('libcloud.common.aws.AWSRequestSignerAlgorithmV4._get_string_to_sign') as mock_get_string:
            with mock.patch('libcloud.common.aws._sign', new=_sign):
                mock_get_key.return_value = 'my_signing_key'
                mock_get_string.return_value = 'my_string_to_sign'
                sig = self.signer._get_signature({}, {}, self.now, method='GET', path='/', data=None)
    self.assertEqual(sig, 'H|my_signing_key|my_string_to_sign')