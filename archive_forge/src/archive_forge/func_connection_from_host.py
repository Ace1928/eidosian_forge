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
def connection_from_host(self, host, port=None, scheme='http', pool_kwargs=None):
    if scheme == 'https':
        return super(ProxyManager, self).connection_from_host(host, port, scheme, pool_kwargs=pool_kwargs)
    return super(ProxyManager, self).connection_from_host(self.proxy.host, self.proxy.port, self.proxy.scheme, pool_kwargs=pool_kwargs)