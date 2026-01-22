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
def block_pipeline_command(name: str) -> Callable[..., Any]:
    """
    Prints error because some pipelined commands should
    be blocked when running in cluster-mode
    """

    def inner(*args, **kwargs):
        raise RedisClusterException(f'ERROR: Calling pipelined function {name} is blocked when running redis in cluster mode...')
    return inner