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
def sentinel_get_master_addr_by_name(self, service_name: str) -> Awaitable:
    """Returns a (host, port) pair for the given ``service_name``"""
    return self.execute_command('SENTINEL GET-MASTER-ADDR-BY-NAME', service_name)