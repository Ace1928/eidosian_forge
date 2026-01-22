import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_key_full_key_value_encoding(self):
    input_dict = {'FirstKey': 'One', 'SecondKey': 'Two'}
    res = self.service_connection._build_tag_list(input_dict)
    expected = {'Tags.member.1.Key': 'FirstKey', 'Tags.member.1.Value': 'One', 'Tags.member.2.Key': 'SecondKey', 'Tags.member.2.Value': 'Two'}
    self.assertEqual(expected, res)