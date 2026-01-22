import logging
import os
import urllib.parse
from oslo_config import cfg
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import units
import requests
from requests import adapters
from requests.packages.urllib3.util import retry
import glance_store
from glance_store import capabilities
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _, _LE
from glance_store import location
def http_response_iterator(conn, response, size):
    """Return an iterator for a file-like object.

    :param conn: HTTP(S) Connection
    :param response: http_client.HTTPResponse object
    :param size: Chunk size to iterate with
    """
    try:
        chunk = response.read(size)
        while chunk:
            yield chunk
            chunk = response.read(size)
    finally:
        conn.close()