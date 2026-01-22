import asyncio
import threading
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, NoReturn, Optional, Union
from aioredis.exceptions import LockError, LockNotOwnedError

        Resets a TTL of an already acquired lock back to a timeout value.
        