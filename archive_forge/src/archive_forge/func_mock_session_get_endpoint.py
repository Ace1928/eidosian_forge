from unittest import mock
from keystoneauth1 import identity
from keystoneauth1 import session
def mock_session_get_endpoint(sess, endpoint_response):
    sess.get_endpoint = mock.MagicMock()
    sess.get_endpoint.return_value = endpoint_response