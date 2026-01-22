import asyncio
import collections
import random
import socket
import ssl
import warnings
from typing import (
from redis._parsers import AsyncCommandsParser, Encoder
from redis._parsers.helpers import (
from redis.asyncio.client import ResponseCallbackT
from redis.asyncio.connection import Connection, DefaultParser, SSLConnection, parse_url
from redis.asyncio.lock import Lock
from redis.asyncio.retry import Retry
from redis.backoff import default_backoff
from redis.client import EMPTY_RESPONSE, NEVER_DECODE, AbstractRedis
from redis.cluster import (
from redis.commands import READ_COMMANDS, AsyncRedisClusterCommands
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.typing import AnyKeyT, EncodableT, KeyT
from redis.utils import (
def get_node_from_key(self, key: str, replica: bool=False) -> Optional['ClusterNode']:
    """
        Get the cluster node corresponding to the provided key.

        :param key:
        :param replica:
            | Indicates if a replica should be returned
            |
              None will returned if no replica holds this key

        :raises SlotNotCoveredError: if the key is not covered by any slot.
        """
    slot = self.keyslot(key)
    slot_cache = self.nodes_manager.slots_cache.get(slot)
    if not slot_cache:
        raise SlotNotCoveredError(f'Slot "{slot}" is not covered by the cluster.')
    if replica:
        if len(self.nodes_manager.slots_cache[slot]) < 2:
            return None
        node_idx = 1
    else:
        node_idx = 0
    return slot_cache[node_idx]