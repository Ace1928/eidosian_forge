from __future__ import absolute_import
import collections
import functools
import logging
from ._collections import RecentlyUsedContainer
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, port_by_scheme
from .exceptions import (
from .packages import six
from .packages.six.moves.urllib.parse import urljoin
from .request import RequestMethods
from .util.proxy import connection_requires_http_tunnel
from .util.retry import Retry
from .util.url import parse_url
def connection_from_context(self, request_context):
    """
        Get a :class:`urllib3.connectionpool.ConnectionPool` based on the request context.

        ``request_context`` must at least contain the ``scheme`` key and its
        value must be a key in ``key_fn_by_scheme`` instance variable.
        """
    scheme = request_context['scheme'].lower()
    pool_key_constructor = self.key_fn_by_scheme.get(scheme)
    if not pool_key_constructor:
        raise URLSchemeUnknown(scheme)
    pool_key = pool_key_constructor(request_context)
    return self.connection_from_pool_key(pool_key, request_context=request_context)