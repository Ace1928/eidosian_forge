import asyncio
import datetime
import hashlib
import inspect
import re
import time as mod_time
import warnings
from typing import (
from aioredis.compat import Protocol, TypedDict
from aioredis.connection import (
from aioredis.exceptions import (
from aioredis.lock import Lock
from aioredis.utils import safe_str, str_if_bytes
def parse_response(self, connection: Connection, command_name: Union[str, bytes], **options):
    result = super().parse_response(connection, command_name, **options)
    if command_name in self.UNWATCH_COMMANDS:
        self.watching = False
    elif command_name == 'WATCH':
        self.watching = True
    return result