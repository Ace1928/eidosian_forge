import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_keyslot(self, key: str) -> ResponseT:
    """
        Returns the hash slot of the specified key
        Sends to random node in the cluster

        For more information see https://redis.io/commands/cluster-keyslot
        """
    return self.execute_command('CLUSTER KEYSLOT', key)