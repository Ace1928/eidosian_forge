import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_canonical_headers_allow_numeric_header_value(self):
    headers = {'Accept-Encoding': 'gzip,deflate', 'Content-Length': 314}
    self.assertEqual(self.signer._get_canonical_headers(headers), 'accept-encoding:gzip,deflate\ncontent-length:314\n')