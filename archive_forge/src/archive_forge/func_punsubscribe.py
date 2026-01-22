import asyncio
import copy
import inspect
import re
import ssl
import warnings
from typing import (
from redis._parsers.helpers import (
from redis.asyncio.connection import (
from redis.asyncio.lock import Lock
from redis.asyncio.retry import Retry
from redis.client import (
from redis.commands import (
from redis.compat import Protocol, TypedDict
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.typing import ChannelT, EncodableT, KeyT
from redis.utils import (
def punsubscribe(self, *args: ChannelT) -> Awaitable:
    """
        Unsubscribe from the supplied patterns. If empty, unsubscribe from
        all patterns.
        """
    patterns: Iterable[ChannelT]
    if args:
        parsed_args = list_or_args((args[0],), args[1:])
        patterns = self._normalize_keys(dict.fromkeys(parsed_args)).keys()
    else:
        parsed_args = []
        patterns = self.patterns
    self.pending_unsubscribe_patterns.update(patterns)
    return self.execute_command('PUNSUBSCRIBE', *parsed_args)