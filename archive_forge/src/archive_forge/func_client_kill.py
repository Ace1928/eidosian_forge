import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_kill(self, address: str, **kwargs) -> ResponseT:
    """Disconnects the client at ``address`` (ip:port)

        For more information see https://redis.io/commands/client-kill
        """
    return self.execute_command('CLIENT KILL', address, **kwargs)