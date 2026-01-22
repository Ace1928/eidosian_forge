from __future__ import annotations
import math
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Mapping
from os import PathLike
from signal import Signals
from socket import AddressFamily, SocketKind, socket
from typing import (
@classmethod
@abstractmethod
def create_capacity_limiter(cls, total_tokens: float) -> CapacityLimiter:
    pass