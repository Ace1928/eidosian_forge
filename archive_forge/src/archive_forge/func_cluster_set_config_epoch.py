import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_set_config_epoch(self, epoch: int, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
    """
        Set the configuration epoch in a new node

        For more information see https://redis.io/commands/cluster-set-config-epoch
        """
    return self.execute_command('CLUSTER SET-CONFIG-EPOCH', epoch, target_nodes=target_nodes)