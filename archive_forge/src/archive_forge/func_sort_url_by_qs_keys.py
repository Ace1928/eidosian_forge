from cryptography import exceptions as crypto_exception
import glance_store as store
from unittest import mock
import urllib
from oslo_config import cfg
from oslo_policy import policy
from glance.async_.flows._internal_plugins import base_download
from glance.common import exception
from glance.common import store_utils
from glance.common import wsgi
import glance.context
import glance.db.simple.api as simple_db
def sort_url_by_qs_keys(url):
    parsed = urllib.parse.urlparse(url)
    queries = urllib.parse.parse_qsl(parsed.query, True)
    sorted_query = sorted(queries, key=lambda x: x[0])
    encoded_sorted_query = urllib.parse.urlencode(sorted_query, True)
    url_parts = (parsed.scheme, parsed.netloc, parsed.path, parsed.params, encoded_sorted_query, parsed.fragment)
    return urllib.parse.urlunparse(url_parts)