import logging
import contextlib
import copy
import time
from asyncio import shield, Event, Future
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Pattern, Set
from aiokafka.errors import IllegalStateError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.abc import ConsumerRebalanceListener
from aiokafka.util import create_future, get_running_loop
def reset_to(self, position: int):
    """ Called by Fetcher after performing a reset to force position to
        a new point.
        """
    assert self._status == PartitionStatus.AWAITING_RESET
    self._position = position
    self._reset_strategy = None
    self._status = PartitionStatus.CONSUMING
    if not self._position_fut.done():
        self._position_fut.set_result(None)