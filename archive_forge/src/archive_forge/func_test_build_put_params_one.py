import datetime
from boto.ec2.cloudwatch import CloudWatchConnection
from tests.compat import unittest, OrderedDict
def test_build_put_params_one(self):
    c = CloudWatchConnection()
    params = {}
    c.build_put_params(params, name='N', value=1, dimensions={'D': 'V'})
    expected_params = {'MetricData.member.1.MetricName': 'N', 'MetricData.member.1.Value': 1, 'MetricData.member.1.Dimensions.member.1.Name': 'D', 'MetricData.member.1.Dimensions.member.1.Value': 'V'}
    self.assertEqual(params, expected_params)