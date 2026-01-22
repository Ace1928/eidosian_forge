import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_canonical_headers_joins_all_headers(self):
    headers = {'accept-encoding': 'gzip,deflate', 'host': 'my_host'}
    self.assertEqual(self.signer._get_canonical_headers(headers), 'accept-encoding:gzip,deflate\nhost:my_host\n')