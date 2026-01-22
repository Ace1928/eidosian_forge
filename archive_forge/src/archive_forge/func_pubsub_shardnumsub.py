import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def pubsub_shardnumsub(self, *args: ChannelT, **kwargs) -> ResponseT:
    """
        Return a list of (shard_channel, number of subscribers) tuples
        for each channel given in ``*args``

        For more information see https://redis.io/commands/pubsub-shardnumsub
        """
    return self.execute_command('PUBSUB SHARDNUMSUB', *args, **kwargs)