from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
def test_cloudsearch_deletion(self):
    """
        Check that the correct arguments are sent to AWS when creating a
        cloudsearch connection.
        """
    self.set_http_response(status_code=200)
    api_response = self.service_connection.delete_domain('demo')
    self.assert_request_parameters({'Action': 'DeleteDomain', 'DomainName': 'demo', 'Version': '2011-02-01'})