import time
import flask  # type: ignore
import pytest  # type: ignore
from pytest_localserver.http import WSGIServer  # type: ignore
from six.moves import http_client
from google.auth import exceptions
def test_request_error(self, server):
    request = self.make_request()
    response = request(url=server.url + '/server_error', method='GET')
    assert response.status == http_client.INTERNAL_SERVER_ERROR
    assert response.data == b'Error'