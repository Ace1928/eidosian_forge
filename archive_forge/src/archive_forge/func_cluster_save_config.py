import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_save_config(self, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
    """
        Forces the node to save cluster state on disk

        For more information see https://redis.io/commands/cluster-saveconfig
        """
    return self.execute_command('CLUSTER SAVECONFIG', target_nodes=target_nodes)