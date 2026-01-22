import copy
import re
import threading
import time
import warnings
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type, Union
from redis._parsers.encoders import Encoder
from redis._parsers.helpers import (
from redis.commands import (
from redis.connection import (
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def ssubscribe(self, *args, target_node=None, **kwargs):
    """
        Subscribes the client to the specified shard channels.
        Channels supplied as keyword arguments expect a channel name as the key
        and a callable as the value. A channel's callable will be invoked automatically
        when a message is received on that channel rather than producing a message via
        ``listen()`` or ``get_sharded_message()``.
        """
    if args:
        args = list_or_args(args[0], args[1:])
    new_s_channels = dict.fromkeys(args)
    new_s_channels.update(kwargs)
    ret_val = self.execute_command('SSUBSCRIBE', *new_s_channels.keys())
    new_s_channels = self._normalize_keys(new_s_channels)
    self.shard_channels.update(new_s_channels)
    if not self.subscribed:
        self.subscribed_event.set()
        self.health_check_response_counter = 0
    self.pending_unsubscribe_shard_channels.difference_update(new_s_channels)
    return ret_val