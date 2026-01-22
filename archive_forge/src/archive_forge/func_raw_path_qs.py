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
@cached_property
def raw_path_qs(self):
    """Encoded path of URL with query."""
    if not self.raw_query_string:
        return self.raw_path
    return f'{self.raw_path}?{self.raw_query_string}'