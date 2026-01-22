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
def wait_for_subscription(self):
    """ Wait for subscription change. This will always wait for next
        subscription.
        """
    fut = create_future()
    self._subscription_waiters.append(fut)
    return fut