import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_describe_cluster(self):
    self.set_http_response(200)
    with self.assertRaises(TypeError):
        self.service_connection.describe_cluster()
    response = self.service_connection.describe_cluster(cluster_id='j-123')
    self.assertTrue(isinstance(response, Cluster))
    self.assertEqual(response.id, 'j-aaaaaaaaa')
    self.assertEqual(response.runningamiversion, '2.4.2')
    self.assertEqual(response.visibletoallusers, 'true')
    self.assertEqual(response.autoterminate, 'false')
    self.assertEqual(response.name, 'test analytics')
    self.assertEqual(response.requestedamiversion, '2.4.2')
    self.assertEqual(response.terminationprotected, 'false')
    self.assertEqual(response.ec2instanceattributes.ec2availabilityzone, 'us-west-1c')
    self.assertEqual(response.ec2instanceattributes.ec2keyname, 'my_secret_key')
    self.assertEqual(response.status.state, 'TERMINATED')
    self.assertEqual(response.applications[0].name, 'hadoop')
    self.assertEqual(response.applications[0].version, '1.0.3')
    self.assertEqual(response.masterpublicdnsname, 'ec2-184-0-0-1.us-west-1.compute.amazonaws.com')
    self.assertEqual(response.normalizedinstancehours, '10')
    self.assertEqual(response.servicerole, 'my-service-role')
    self.assert_request_parameters({'Action': 'DescribeCluster', 'ClusterId': 'j-123', 'Version': '2009-03-31'})