from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
def test_cloudsearch_searchservice_creation(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_domain('demo')
    domain = Domain(self, api_response)
    search = domain.get_search_service()
    self.assertEqual(search.endpoint, 'search-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')