import os
import socket
from tests.compat import mock, unittest
from httpretty import HTTPretty
from boto import UserAgent
from boto.compat import json, parse_qs
from boto.connection import AWSQueryConnection, AWSAuthConnection, HTTPRequest
from boto.exception import BotoServerError
from boto.regioninfo import RegionInfo
@mock.patch.object(socket, 'create_connection')
@mock.patch('boto.compat.http_client.HTTPResponse')
@mock.patch('boto.connection.ssl', autospec=True)
def test_proxy_ssl_with_verification(self, ssl_mock, http_response_mock, create_connection_mock):
    type(http_response_mock.return_value).status = mock.PropertyMock(return_value=200)
    conn = AWSAuthConnection('mockservice.s3.amazonaws.com', aws_access_key_id='access_key', aws_secret_access_key='secret', suppress_consec_slashes=False, proxy_port=80)
    conn.https_validate_certificates = True
    dummy_cert = {'subjectAltName': (('DNS', 's3.amazonaws.com'), ('DNS', '*.s3.amazonaws.com'))}
    mock_sock = mock.Mock()
    create_connection_mock.return_value = mock_sock
    mock_sslSock = mock.Mock()
    mock_sslSock.getpeercert.return_value = dummy_cert
    mock_context = mock.Mock()
    mock_context.wrap_socket.return_value = mock_sslSock
    ssl_mock.create_default_context.return_value = mock_context
    conn.proxy_ssl('mockservice.s3.amazonaws.com', 80)
    mock_sslSock.getpeercert.assert_called_once_with()
    mock_context.wrap_socket.assert_called_once_with(mock_sock, server_hostname='mockservice.s3.amazonaws.com')