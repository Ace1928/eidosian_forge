import random
import socket
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from redis._parsers import CommandsParser, Encoder
from redis._parsers.helpers import parse_scan
from redis.backoff import default_backoff
from redis.client import CaseInsensitiveDict, PubSub, Redis
from redis.commands import READ_COMMANDS, RedisClusterCommands
from redis.commands.helpers import list_or_args
from redis.connection import ConnectionPool, DefaultParser, parse_url
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def get_sharded_message(self, ignore_subscribe_messages=False, timeout=0.0, target_node=None):
    if target_node:
        message = self.node_pubsub_mapping[target_node.name].get_message(ignore_subscribe_messages=ignore_subscribe_messages, timeout=timeout)
    else:
        message = self._sharded_message_generator()
    if message is None:
        return None
    elif str_if_bytes(message['type']) == 'sunsubscribe':
        if message['channel'] in self.pending_unsubscribe_shard_channels:
            self.pending_unsubscribe_shard_channels.remove(message['channel'])
            self.shard_channels.pop(message['channel'], None)
            node = self.cluster.get_node_from_key(message['channel'])
            if self.node_pubsub_mapping[node.name].subscribed is False:
                self.node_pubsub_mapping.pop(node.name)
    if not self.channels and (not self.patterns) and (not self.shard_channels):
        self.subscribed_event.clear()
    if self.ignore_subscribe_messages or ignore_subscribe_messages:
        return None
    return message