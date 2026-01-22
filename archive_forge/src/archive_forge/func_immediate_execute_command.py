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
def immediate_execute_command(self, *args, **options):
    """
        Execute a command immediately, but don't auto-retry on a
        ConnectionError if we're already WATCHing a variable. Used when
        issuing WATCH or subsequent commands retrieving their values but before
        MULTI is called.
        """
    command_name = args[0]
    conn = self.connection
    if not conn:
        conn = self.connection_pool.get_connection(command_name, self.shard_hint)
        self.connection = conn
    return conn.retry.call_with_retry(lambda: self._send_command_parse_response(conn, command_name, *args, **options), lambda error: self._disconnect_reset_raise(conn, error))