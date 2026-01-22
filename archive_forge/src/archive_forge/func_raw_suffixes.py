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
def raw_suffixes(self):
    name = self.raw_name
    if name.endswith('.'):
        return ()
    name = name.lstrip('.')
    return tuple(('.' + suffix for suffix in name.split('.')[1:]))