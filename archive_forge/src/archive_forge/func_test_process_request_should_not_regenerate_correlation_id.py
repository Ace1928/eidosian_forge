from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_middleware import correlation_id
def test_process_request_should_not_regenerate_correlation_id(self):
    app = mock.Mock()
    req = mock.Mock()
    req.headers = {'X_CORRELATION_ID': 'correlation_id'}
    middleware = correlation_id.CorrelationId(app)
    middleware(req)
    self.assertEqual('correlation_id', req.headers.get('X_CORRELATION_ID'))