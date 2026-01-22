import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def migrate(self, host: str, port: int, keys: KeysT, destination_db: int, timeout: int, copy: bool=False, replace: bool=False, auth: Union[str, None]=None, **kwargs) -> ResponseT:
    """
        Migrate 1 or more keys from the current Redis server to a different
        server specified by the ``host``, ``port`` and ``destination_db``.

        The ``timeout``, specified in milliseconds, indicates the maximum
        time the connection between the two servers can be idle before the
        command is interrupted.

        If ``copy`` is True, the specified ``keys`` are NOT deleted from
        the source server.

        If ``replace`` is True, this operation will overwrite the keys
        on the destination server if they exist.

        If ``auth`` is specified, authenticate to the destination server with
        the password provided.

        For more information see https://redis.io/commands/migrate
        """
    keys = list_or_args(keys, [])
    if not keys:
        raise DataError('MIGRATE requires at least one key')
    pieces = []
    if copy:
        pieces.append(b'COPY')
    if replace:
        pieces.append(b'REPLACE')
    if auth:
        pieces.append(b'AUTH')
        pieces.append(auth)
    pieces.append(b'KEYS')
    pieces.extend(keys)
    return self.execute_command('MIGRATE', host, port, '', destination_db, timeout, *pieces, **kwargs)