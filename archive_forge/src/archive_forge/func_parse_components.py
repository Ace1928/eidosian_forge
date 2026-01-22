import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def parse_components(url):
    parsed = parse.urlsplit(url)
    query = parse.parse_qs(parsed.query)
    return (parsed._replace(query=''), query)