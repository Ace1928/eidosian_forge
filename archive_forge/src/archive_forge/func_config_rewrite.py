import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def config_rewrite(self, **kwargs) -> ResponseT:
    """
        Rewrite config file with the minimal change to reflect running config.

        For more information see https://redis.io/commands/config-rewrite
        """
    return self.execute_command('CONFIG REWRITE', **kwargs)