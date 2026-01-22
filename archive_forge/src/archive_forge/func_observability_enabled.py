from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
@property
def observability_enabled(self) -> bool:
    return self.tracing_enabled or self.stats_enabled