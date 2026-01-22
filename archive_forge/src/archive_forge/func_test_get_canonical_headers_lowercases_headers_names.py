import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_canonical_headers_lowercases_headers_names(self):
    headers = {'Accept-Encoding': 'GZIP,DEFLATE', 'User-Agent': 'My-UA'}
    self.assertEqual(self.signer._get_canonical_headers(headers), 'accept-encoding:GZIP,DEFLATE\nuser-agent:My-UA\n')