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
def assign_from_subscribed(self, assignment: Set[TopicPartition]):
    """ Set assignment if automatic assignment is used.

        Caller: Coordinator
        Affects: SubscriptionState.subscription.assignment
        """
    assert self._subscription_type in [SubscriptionType.AUTO_PATTERN, SubscriptionType.AUTO_TOPICS]
    self._subscription._assign(assignment)
    self._notify_assignment_waiters()