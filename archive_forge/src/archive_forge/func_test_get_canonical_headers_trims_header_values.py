import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_canonical_headers_trims_header_values(self):
    headers = {'accept-encoding': '   gzip,deflate', 'user-agent': 'libcloud/0.17.0 '}
    self.assertEqual(self.signer._get_canonical_headers(headers), 'accept-encoding:gzip,deflate\nuser-agent:libcloud/0.17.0\n')