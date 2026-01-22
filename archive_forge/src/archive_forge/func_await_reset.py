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
def await_reset(self, strategy):
    """ Called by either Coonsumer in `seek_to_*` or by Coordinator after
        setting initial committed point.
        """
    self._reset_strategy = strategy
    self._position = None
    if self._position_fut.done():
        self._position_fut = create_future()
    self._status = PartitionStatus.AWAITING_RESET