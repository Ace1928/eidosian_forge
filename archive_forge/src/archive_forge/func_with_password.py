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
def with_password(self, password):
    """Return a new URL with password replaced.

        Autoencode password if needed.

        Clear password if argument is None.

        """
    if password is None:
        pass
    elif isinstance(password, str):
        password = self._QUOTER(password)
    else:
        raise TypeError('Invalid password type')
    if not self.is_absolute():
        raise ValueError('password replacement is not allowed for relative URLs')
    val = self._val
    return URL(self._val._replace(netloc=self._make_netloc(val.username, password, val.hostname, val.port)), encoded=True)