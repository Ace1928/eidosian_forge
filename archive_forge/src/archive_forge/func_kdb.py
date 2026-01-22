from __future__ import annotations
import json
import asyncio
import contextlib
from lazyops.types.lazydict import LazyDict
from lazyops.utils.times import Timer
from lazyops.utils.logs import Logger
from lazyops.utils.helpers import create_timestamp, adeferred_task
from lazyops.utils.pooler import ThreadPooler
from proposalflow.utils.logs import logger, null_logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple, Union, Set, TYPE_CHECKING
@property
def kdb(self) -> 'KVDBSession':
    """
        Gets the keydb
        """
    if self._kdb is None:
        self._kdb = self.settings.ctx.get_kdb_session(self.name)
    return self._kdb