import ssl
from . import http_server, ssl_certs, test_server
Verify the request.

        Return True if we should proceed with this request, False if we should
        not even touch a single byte in the socket !
        