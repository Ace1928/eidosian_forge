from __future__ import annotations
import os
import abc
import atexit
import pathlib
import filelock
import contextlib
from lazyops.types import BaseModel, Field
from lazyops.utils.logs import logger
from lazyops.utils.serialization import Json
from typing import Optional, Dict, Any, Set, List, Union, Generator, TYPE_CHECKING
@property
def is_leader_process(self) -> bool:
    """
        Returns whether this is the leader process
        """
    return self.process_id in self.stx.get('leader_process_ids', []) or self.is_primary_process