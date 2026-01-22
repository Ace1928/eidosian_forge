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
def setup_process_pool_exit_at_shutdown(cls, workers: set[Process]) -> None:
    pass