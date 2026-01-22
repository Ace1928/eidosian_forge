import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_addslotsrange(self, target_node: 'TargetNodesT', *slots: EncodableT) -> ResponseT:
    """
        Similar to the CLUSTER ADDSLOTS command.
        The difference between the two commands is that ADDSLOTS takes a list of slots
        to assign to the node, while ADDSLOTSRANGE takes a list of slot ranges
        (specified by start and end slots) to assign to the node.

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information see https://redis.io/commands/cluster-addslotsrange
        """
    return self.execute_command('CLUSTER ADDSLOTSRANGE', *slots, target_nodes=target_node)