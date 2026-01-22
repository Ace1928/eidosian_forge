import sys
import unittest
from datetime import datetime
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.aws import UNSIGNED_PAYLOAD, SignedAWSConnection, AWSRequestSignerAlgorithmV4
def test_post_canonical_request(self):
    req = self.signer._get_canonical_request({'Action': 'DescribeInstances', 'Version': '2013-10-15'}, {'Accept-Encoding': 'gzip,deflate', 'User-Agent': 'My-UA'}, method='POST', path='/my_action/', data='{}')
    self.assertEqual(req, 'POST\n/my_action/\nAction=DescribeInstances&Version=2013-10-15\naccept-encoding:gzip,deflate\nuser-agent:My-UA\n\naccept-encoding;user-agent\n44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a')