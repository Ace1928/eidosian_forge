import socket
import re
import logging
import warnings
from requests.exceptions import RequestException, SSLError
import http.client as http_client
from urllib.parse import quote, unquote
from urllib.parse import urljoin, urlparse, urlunparse
from time import sleep, time
from swiftclient import version as swiftclient_version
from swiftclient.exceptions import ClientException
from swiftclient.requests_compat import SwiftClientRequestsSession
from swiftclient.utils import (
def resp_header_dict(resp):
    resp_headers = {}
    for header, value in resp.getheaders():
        header = parse_header_string(header).lower()
        resp_headers[header] = parse_header_string(value)
    return resp_headers