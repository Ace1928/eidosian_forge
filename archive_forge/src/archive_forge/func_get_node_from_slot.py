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
def get_node_from_slot(self, slot: int, read_from_replicas: bool=False) -> 'ClusterNode':
    if self._moved_exception:
        self._update_moved_slots()
    try:
        if read_from_replicas:
            primary_name = self.slots_cache[slot][0].name
            node_idx = self.read_load_balancer.get_server_index(primary_name, len(self.slots_cache[slot]))
            return self.slots_cache[slot][node_idx]
        return self.slots_cache[slot][0]
    except (IndexError, TypeError):
        raise SlotNotCoveredError(f'Slot "{slot}" not covered by the cluster. "require_full_coverage={self.require_full_coverage}"')