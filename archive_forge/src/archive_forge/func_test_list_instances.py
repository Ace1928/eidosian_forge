import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_list_instances(self):
    self.set_http_response(200)
    with self.assertRaises(TypeError):
        self.service_connection.list_instances()
    response = self.service_connection.list_instances(cluster_id='j-123')
    self.assertTrue(isinstance(response, InstanceList))
    self.assertEqual(len(response.instances), 3)
    self.assertTrue(isinstance(response.instances[0], InstanceInfo))
    self.assertEqual(response.instances[0].ec2instanceid, 'i-aaaaaaaa')
    self.assertEqual(response.instances[0].id, 'ci-123456789abc')
    self.assertEqual(response.instances[0].privatednsname, 'ip-10-0-0-60.us-west-1.compute.internal')
    self.assertEqual(response.instances[0].privateipaddress, '10.0.0.60')
    self.assertEqual(response.instances[0].publicdnsname, 'ec2-54-0-0-1.us-west-1.compute.amazonaws.com')
    self.assertEqual(response.instances[0].publicipaddress, '54.0.0.1')
    self.assert_request_parameters({'Action': 'ListInstances', 'ClusterId': 'j-123', 'Version': '2009-03-31'})