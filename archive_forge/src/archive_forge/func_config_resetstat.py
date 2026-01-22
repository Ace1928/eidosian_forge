import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def config_resetstat(self, **kwargs) -> ResponseT:
    """
        Reset runtime statistics

        For more information see https://redis.io/commands/config-resetstat
        """
    return self.execute_command('CONFIG RESETSTAT', **kwargs)