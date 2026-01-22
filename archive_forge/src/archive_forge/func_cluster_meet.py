import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_meet(self, host: str, port: int, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
    """
        Force a node cluster to handshake with another node.
        Sends to specified node.

        For more information see https://redis.io/commands/cluster-meet
        """
    return self.execute_command('CLUSTER MEET', host, port, target_nodes=target_nodes)