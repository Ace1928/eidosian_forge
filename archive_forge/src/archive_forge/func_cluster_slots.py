import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_slots(self, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
    """
        Get array of Cluster slot to node mappings

        For more information see https://redis.io/commands/cluster-slots
        """
    return self.execute_command('CLUSTER SLOTS', target_nodes=target_nodes)