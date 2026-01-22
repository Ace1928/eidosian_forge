import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
def prepare_request_dict(request_dict, endpoint_url, context=None, user_agent=None):
    """
    This method prepares a request dict to be created into an
    AWSRequestObject. This prepares the request dict by adding the
    url and the user agent to the request dict.

    :type request_dict: dict
    :param request_dict:  The request dict (created from the
        ``serialize`` module).

    :type user_agent: string
    :param user_agent: The user agent to use for this request.

    :type endpoint_url: string
    :param endpoint_url: The full endpoint url, which contains at least
        the scheme, the hostname, and optionally any path components.
    """
    r = request_dict
    if user_agent is not None:
        headers = r['headers']
        headers['User-Agent'] = user_agent
    host_prefix = r.get('host_prefix')
    url = _urljoin(endpoint_url, r['url_path'], host_prefix)
    if r['query_string']:
        percent_encode_sequence = botocore.utils.percent_encode_sequence
        encoded_query_string = percent_encode_sequence(r['query_string'])
        if '?' not in url:
            url += '?%s' % encoded_query_string
        else:
            url += '&%s' % encoded_query_string
    r['url'] = url
    r['context'] = context
    if context is None:
        r['context'] = {}