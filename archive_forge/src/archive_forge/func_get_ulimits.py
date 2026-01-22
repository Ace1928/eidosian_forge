import hashlib
import inspect
import logging
from contextlib import contextmanager, asynccontextmanager
from typing import Union, Optional, List, Callable, Generator, AsyncGenerator, TYPE_CHECKING
from aiokeydb.v2.types import ENOVAL
from redis.utils import (
def get_ulimits():
    import resource
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft_limit