from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
def test_cloudsearch_index_documents_resp(self):
    """
        Check that the AWS response is being parsed correctly when indexing a
        domain.
        """
    self.set_http_response(status_code=200)
    api_response = self.service_connection.index_documents('demo')
    self.assertEqual(api_response, ['average_score', 'brand_id', 'colors', 'context', 'context_owner', 'created_at', 'creator_id', 'description', 'file_size', 'format', 'has_logo', 'has_messaging', 'height', 'image_id', 'ingested_from', 'is_advertising', 'is_photo', 'is_reviewed', 'modified_at', 'subject_date', 'tags', 'title', 'width'])