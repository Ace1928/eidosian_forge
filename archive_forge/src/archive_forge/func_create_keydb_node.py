import copy
import random
import socket
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union
from aiokeydb.v1.core import CaseInsensitiveDict, PubSub, KeyDB, parse_scan
from aiokeydb.v1.commands import READ_COMMANDS, CommandsParser, KeyDBClusterCommands
from aiokeydb.v1.connection import ConnectionPool, DefaultParser, Encoder, parse_url
from aiokeydb.v1.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.lock import Lock
from aiokeydb.v1.utils import (
def create_keydb_node(self, host, port, **kwargs):
    if self.from_url:
        kwargs.update({'host': host})
        kwargs.update({'port': port})
        r = KeyDB(connection_pool=ConnectionPool(**kwargs))
    else:
        r = KeyDB(host=host, port=port, **kwargs)
    return r