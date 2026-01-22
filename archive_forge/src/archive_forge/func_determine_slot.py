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
def determine_slot(self, *args):
    """
        Figure out what slot to use based on args.

        Raises a RedisClusterException if there's a missing key and we can't
            determine what slots to map the command to; or, if the keys don't
            all map to the same key slot.
        """
    command = args[0]
    if self.command_flags.get(command) == SLOT_ID:
        return args[1]
    if command.upper() in ('EVAL', 'EVALSHA'):
        if len(args) <= 2:
            raise RedisClusterException(f'Invalid args in command: {args}')
        num_actual_keys = int(args[2])
        eval_keys = args[3:3 + num_actual_keys]
        if len(eval_keys) == 0:
            return random.randrange(0, REDIS_CLUSTER_HASH_SLOTS)
        keys = eval_keys
    else:
        keys = self._get_command_keys(*args)
        if keys is None or len(keys) == 0:
            if command.upper() in ('FCALL', 'FCALL_RO'):
                return random.randrange(0, REDIS_CLUSTER_HASH_SLOTS)
            raise RedisClusterException(f'No way to dispatch this command to Redis Cluster. Missing key.\nYou can execute the command by specifying target nodes.\nCommand: {args}')
    if len(keys) == 1:
        return self.keyslot(keys[0])
    slots = {self.keyslot(key) for key in keys}
    if len(slots) != 1:
        raise RedisClusterException(f'{command} - all keys must map to the same key slot')
    return slots.pop()