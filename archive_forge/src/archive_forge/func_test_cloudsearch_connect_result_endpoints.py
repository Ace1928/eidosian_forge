from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
def test_cloudsearch_connect_result_endpoints(self):
    """Check that endpoints & ARNs are correctly returned from AWS"""
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_domain('demo')
    domain = Domain(self, api_response)
    self.assertEqual(domain.doc_service_arn, 'arn:aws:cs:us-east-1:1234567890:doc/demo')
    self.assertEqual(domain.doc_service_endpoint, 'doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')
    self.assertEqual(domain.search_service_arn, 'arn:aws:cs:us-east-1:1234567890:search/demo')
    self.assertEqual(domain.search_service_endpoint, 'search-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')