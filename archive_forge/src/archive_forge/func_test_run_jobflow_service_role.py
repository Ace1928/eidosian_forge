import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_run_jobflow_service_role(self):
    self.set_http_response(200)
    response = self.service_connection.run_jobflow('EmrCluster', service_role='EMR_DefaultRole')
    self.assertTrue(response)
    self.assert_request_parameters({'Action': 'RunJobFlow', 'Version': '2009-03-31', 'ServiceRole': 'EMR_DefaultRole', 'Name': 'EmrCluster'}, ignore_params_values=['ActionOnFailure', 'Instances.InstanceCount', 'Instances.KeepJobFlowAliveWhenNoSteps', 'Instances.MasterInstanceType', 'Instances.SlaveInstanceType'])