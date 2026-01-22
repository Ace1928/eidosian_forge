import functools
import math
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from ipaddress import ip_address
from urllib.parse import SplitResult, parse_qsl, quote, urljoin, urlsplit, urlunsplit
import idna
from multidict import MultiDict, MultiDictProxy
from ._quoting import _Quoter, _Unquoter
def human_repr(self):
    """Return decoded human readable string for URL representation."""
    user = _human_quote(self.user, '#/:?@[]')
    password = _human_quote(self.password, '#/:?@[]')
    host = self.host
    if host:
        host = self._encode_host(self.host, human=True)
    path = _human_quote(self.path, '#?')
    query_string = '&'.join(('{}={}'.format(_human_quote(k, '#&+;='), _human_quote(v, '#&+;=')) for k, v in self.query.items()))
    fragment = _human_quote(self.fragment, '')
    return urlunsplit(SplitResult(self.scheme, self._make_netloc(user, password, host, self._val.port, encode_host=False), path, query_string, fragment))