import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_describe_jobflows_filtered(self):
    self.set_http_response(200)
    now = datetime.now()
    a_bit_before = datetime.fromtimestamp(time() - 1000)
    self.service_connection.describe_jobflows(states=['WAITING', 'RUNNING'], jobflow_ids=['j-aaaaaa', 'j-aaaaab'], created_after=a_bit_before, created_before=now)
    self.assert_request_parameters({'Action': 'DescribeJobFlows', 'JobFlowIds.member.1': 'j-aaaaaa', 'JobFlowIds.member.2': 'j-aaaaab', 'JobFlowStates.member.1': 'WAITING', 'JobFlowStates.member.2': 'RUNNING', 'CreatedAfter': a_bit_before.strftime(boto.utils.ISO8601), 'CreatedBefore': now.strftime(boto.utils.ISO8601)}, ignore_params_values=['Version'])