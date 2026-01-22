import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def pubsub_channels(self, pattern: PatternT='*', **kwargs) -> ResponseT:
    """
        Return a list of channels that have at least one subscriber

        For more information see https://redis.io/commands/pubsub-channels
        """
    return self.execute_command('PUBSUB CHANNELS', pattern, **kwargs)