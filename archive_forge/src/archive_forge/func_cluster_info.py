import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_info(self, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
    """
        Provides info about Redis Cluster node state.
        The command will be sent to a random node in the cluster if no target
        node is specified.

        For more information see https://redis.io/commands/cluster-info
        """
    return self.execute_command('CLUSTER INFO', target_nodes=target_nodes)