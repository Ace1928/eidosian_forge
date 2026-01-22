import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_run_jobflow_enable_debugging(self):
    self.region = 'ap-northeast-2'
    self.set_http_response(200)
    self.service_connection.run_jobflow('EmrCluster', enable_debugging=True)
    actual_params = set(self.actual_request.params.copy().items())
    expected_params = set([('Steps.member.1.HadoopJarStep.Jar', 's3://ap-northeast-2.elasticmapreduce/libs/script-runner/script-runner.jar'), ('Steps.member.1.HadoopJarStep.Args.member.1', 's3://ap-northeast-2.elasticmapreduce/libs/state-pusher/0.1/fetch')])
    self.assertTrue(expected_params <= actual_params)