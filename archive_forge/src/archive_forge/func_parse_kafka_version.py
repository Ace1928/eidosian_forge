import asyncio
import os
import weakref
from asyncio import AbstractEventLoop
from types import MethodType
from typing import Any, Awaitable, Coroutine, Dict, Tuple, TypeVar, Union, cast
import async_timeout
from packaging.version import Version
from .structs import OffsetAndMetadata, TopicPartition
def parse_kafka_version(api_version: str) -> Tuple[int, int, int]:
    parsed = Version(api_version).release
    if not 2 <= len(parsed) <= 3:
        raise ValueError(api_version)
    version = cast(Tuple[int, int, int], (parsed + (0,))[:3])
    if not (0, 9) <= version < (3, 0):
        raise ValueError(api_version)
    return version