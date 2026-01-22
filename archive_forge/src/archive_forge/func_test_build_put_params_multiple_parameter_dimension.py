import datetime
from boto.ec2.cloudwatch import CloudWatchConnection
from tests.compat import unittest, OrderedDict
def test_build_put_params_multiple_parameter_dimension(self):
    self.maxDiff = None
    c = CloudWatchConnection()
    params = {}
    dimensions = [OrderedDict((('D1', 'V'), ('D2', 'W')))]
    c.build_put_params(params, name='N', value=[1], dimensions=dimensions)
    expected_params = {'MetricData.member.1.MetricName': 'N', 'MetricData.member.1.Value': 1, 'MetricData.member.1.Dimensions.member.1.Name': 'D1', 'MetricData.member.1.Dimensions.member.1.Value': 'V', 'MetricData.member.1.Dimensions.member.2.Name': 'D2', 'MetricData.member.1.Dimensions.member.2.Value': 'W'}
    self.assertEqual(params, expected_params)