import http.client
import http.server
import threading
from oslo_utils import units
def serve_requests(httpd):
    httpd.serve_forever()