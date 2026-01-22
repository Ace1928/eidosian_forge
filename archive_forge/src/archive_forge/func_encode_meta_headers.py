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
def encode_meta_headers(headers):
    """Only encode metadata headers keys"""
    ret = {}
    for header, value in headers.items():
        value = encode_utf8(value)
        header = header.lower()
        if isinstance(header, str) and header.startswith(USER_METADATA_TYPE):
            header = encode_utf8(header)
        ret[header] = value
    return ret