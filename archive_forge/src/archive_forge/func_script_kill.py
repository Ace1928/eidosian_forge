import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def script_kill(self) -> ResponseT:
    """
        Kill the currently executing Lua script

        For more information see https://redis.io/commands/script-kill
        """
    return self.execute_command('SCRIPT KILL')