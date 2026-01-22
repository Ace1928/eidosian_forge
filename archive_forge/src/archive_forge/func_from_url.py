import asyncio
import copy
import inspect
import re
import ssl
import warnings
from typing import (
from redis._parsers.helpers import (
from redis.asyncio.connection import (
from redis.asyncio.lock import Lock
from redis.asyncio.retry import Retry
from redis.client import (
from redis.commands import (
from redis.compat import Protocol, TypedDict
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.typing import ChannelT, EncodableT, KeyT
from redis.utils import (
@classmethod
def from_url(cls, url: str, single_connection_client: bool=False, auto_close_connection_pool: Optional[bool]=None, **kwargs):
    """
        Return a Redis client object configured from the given URL

        For example::

            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0
            unix://[username@]/path/to/socket.sock?db=0[&password=password]

        Three URL schemes are supported:

        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>
        - ``unix://``: creates a Unix Domain Socket connection.

        The username, password, hostname, path and all querystring values
        are passed through urllib.parse.unquote in order to replace any
        percent-encoded values with their corresponding characters.

        There are several ways to specify a database number. The first value
        found will be used:

        1. A ``db`` querystring option, e.g. redis://localhost?db=0

        2. If using the redis:// or rediss:// schemes, the path argument
               of the url, e.g. redis://localhost/0

        3. A ``db`` keyword argument to this function.

        If none of these options are specified, the default db=0 is used.

        All querystring options are cast to their appropriate Python types.
        Boolean arguments can be specified with string values "True"/"False"
        or "Yes"/"No". Values that cannot be properly cast cause a
        ``ValueError`` to be raised. Once parsed, the querystring arguments
        and keyword arguments are passed to the ``ConnectionPool``'s
        class initializer. In the case of conflicting arguments, querystring
        arguments always win.

        """
    connection_pool = ConnectionPool.from_url(url, **kwargs)
    client = cls(connection_pool=connection_pool, single_connection_client=single_connection_client)
    if auto_close_connection_pool is not None:
        warnings.warn(DeprecationWarning('"auto_close_connection_pool" is deprecated since version 5.0.1. Please create a ConnectionPool explicitly and provide to the Redis() constructor instead.'))
    else:
        auto_close_connection_pool = True
    client.auto_close_connection_pool = auto_close_connection_pool
    return client