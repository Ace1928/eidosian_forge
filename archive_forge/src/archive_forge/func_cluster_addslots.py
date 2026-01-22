import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_addslots(self, target_node: 'TargetNodesT', *slots: EncodableT) -> ResponseT:
    """
        Assign new hash slots to receiving node. Sends to specified node.

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information see https://redis.io/commands/cluster-addslots
        """
    return self.execute_command('CLUSTER ADDSLOTS', *slots, target_nodes=target_node)