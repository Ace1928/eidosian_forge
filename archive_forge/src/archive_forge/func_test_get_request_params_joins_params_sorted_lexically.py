import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_get_request_params_joins_params_sorted_lexically(self):
    self.assertEqual(self.signer._get_request_params({'Action': 'DescribeInstances', 'Filter.1.Name': 'state', 'Version': '2013-10-15'}), 'Action=DescribeInstances&Filter.1.Name=state&Version=2013-10-15')