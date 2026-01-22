import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_list_steps_with_states(self):
    self.set_http_response(200)
    response = self.service_connection.list_steps(cluster_id='j-123', step_states=['COMPLETED', 'FAILED'])
    self.assert_request_parameters({'Action': 'ListSteps', 'ClusterId': 'j-123', 'StepStates.member.1': 'COMPLETED', 'StepStates.member.2': 'FAILED', 'Version': '2009-03-31'})
    self.assertTrue(isinstance(response, StepSummaryList))
    self.assertEqual(response.steps[0].name, 'Step 1')