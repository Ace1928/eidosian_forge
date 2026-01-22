from __future__ import annotations
import collections.abc as collections_abc
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.parse import unquote
from .interfaces import Dialect
from .. import exc
from .. import util
from ..dialects import plugins
from ..dialects import registry
def update_query_pairs(self, key_value_pairs: Iterable[Tuple[str, Union[str, List[str]]]], append: bool=False) -> URL:
    """Return a new :class:`_engine.URL` object with the
        :attr:`_engine.URL.query`
        parameter dictionary updated by the given sequence of key/value pairs

        E.g.::

            >>> from sqlalchemy.engine import make_url
            >>> url = make_url("postgresql+psycopg2://user:pass@host/dbname")
            >>> url = url.update_query_pairs([("alt_host", "host1"), ("alt_host", "host2"), ("ssl_cipher", "/path/to/crt")])
            >>> str(url)
            'postgresql+psycopg2://user:pass@host/dbname?alt_host=host1&alt_host=host2&ssl_cipher=%2Fpath%2Fto%2Fcrt'

        :param key_value_pairs: A sequence of tuples containing two strings
         each.

        :param append: if True, parameters in the existing query string will
         not be removed; new parameters will be in addition to those present.
         If left at its default of False, keys present in the given query
         parameters will replace those of the existing query string.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_engine.URL.query`

            :meth:`_engine.URL.difference_update_query`

            :meth:`_engine.URL.set`

        """
    existing_query = self.query
    new_keys: Dict[str, Union[str, List[str]]] = {}
    for key, value in key_value_pairs:
        if key in new_keys:
            new_keys[key] = util.to_list(new_keys[key])
            cast('List[str]', new_keys[key]).append(cast(str, value))
        else:
            new_keys[key] = list(value) if isinstance(value, (list, tuple)) else value
    new_query: Mapping[str, Union[str, Sequence[str]]]
    if append:
        new_query = {}
        for k in new_keys:
            if k in existing_query:
                new_query[k] = tuple(util.to_list(existing_query[k]) + util.to_list(new_keys[k]))
            else:
                new_query[k] = new_keys[k]
        new_query.update({k: existing_query[k] for k in set(existing_query).difference(new_keys)})
    else:
        new_query = self.query.union({k: tuple(v) if isinstance(v, list) else v for k, v in new_keys.items()})
    return self.set(query=new_query)