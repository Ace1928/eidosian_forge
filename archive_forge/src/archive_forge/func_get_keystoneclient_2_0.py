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
def get_keystoneclient_2_0(auth_url, user, key, os_options, **kwargs):
    kwargs.update({'auth_version': '2.0'})
    return get_auth_keystone(auth_url, user, key, os_options, **kwargs)