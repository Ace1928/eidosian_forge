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
def psubscribe(self, *args, **kwargs):
    """
        Subscribe to channel patterns. Patterns supplied as keyword arguments
        expect a pattern name as the key and a callable as the value. A
        pattern's callable will be invoked automatically when a message is
        received on that pattern rather than producing a message via
        ``listen()``.
        """
    if args:
        args = list_or_args(args[0], args[1:])
    new_patterns = dict.fromkeys(args)
    new_patterns.update(kwargs)
    ret_val = self.execute_command('PSUBSCRIBE', *new_patterns.keys())
    new_patterns = self._normalize_keys(new_patterns)
    self.patterns.update(new_patterns)
    if not self.subscribed:
        self.subscribed_event.set()
        self.health_check_response_counter = 0
    self.pending_unsubscribe_patterns.difference_update(new_patterns)
    return ret_val