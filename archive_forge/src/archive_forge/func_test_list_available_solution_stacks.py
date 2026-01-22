import json
from tests.unit import AWSMockServiceTestCase
from boto.beanstalk.layer1 import Layer1
def test_list_available_solution_stacks(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.list_available_solution_stacks()
    stack_details = api_response['ListAvailableSolutionStacksResponse']['ListAvailableSolutionStacksResult']['SolutionStackDetails']
    solution_stacks = api_response['ListAvailableSolutionStacksResponse']['ListAvailableSolutionStacksResult']['SolutionStacks']
    self.assertEqual(solution_stacks, [u'32bit Amazon Linux running Tomcat 7', u'32bit Amazon Linux running PHP 5.3'])
    self.assert_request_parameters({'Action': 'ListAvailableSolutionStacks', 'ContentType': 'JSON', 'Version': '2010-12-01'})