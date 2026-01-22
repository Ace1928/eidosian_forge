from __future__ import annotations
import typing as t
import warnings
from pprint import pformat
from threading import Lock
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlunsplit
from .._internal import _get_environ
from .._internal import _wsgi_decoding_dance
from ..datastructures import ImmutableDict
from ..datastructures import MultiDict
from ..exceptions import BadHost
from ..exceptions import HTTPException
from ..exceptions import MethodNotAllowed
from ..exceptions import NotFound
from ..urls import _urlencode
from ..wsgi import get_host
from .converters import DEFAULT_CONVERTERS
from .exceptions import BuildError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .exceptions import RequestRedirect
from .exceptions import WebsocketMismatch
from .matcher import StateMachineMatcher
from .rules import _simple_rule_re
from .rules import Rule
def make_alias_redirect_url(self, path: str, endpoint: str, values: t.Mapping[str, t.Any], method: str, query_args: t.Mapping[str, t.Any] | str) -> str:
    """Internally called to make an alias redirect URL."""
    url = self.build(endpoint, values, method, append_unknown=False, force_external=True)
    if query_args:
        url += f'?{self.encode_query_args(query_args)}'
    assert url != path, 'detected invalid alias setting. No canonical URL found'
    return url