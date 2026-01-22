import asyncio
import logging
import re
import sys
import traceback
import warnings
from typing import Dict, List
from aiokafka.abc import ConsumerRebalanceListener
from aiokafka.client import AIOKafkaClient
from aiokafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from aiokafka.errors import (
from aiokafka.structs import TopicPartition, ConsumerRecord
from aiokafka.util import (
from aiokafka import __version__
from .fetcher import Fetcher, OffsetResetStrategy
from .group_coordinator import GroupCoordinator, NoGroupCoordinator
from .subscription_state import SubscriptionState
def partitions_for_topic(self, topic):
    """ Get metadata about the partitions for a given topic.

        This method will return `None` if Consumer does not already have
        metadata for this topic.

        Arguments:
            topic (str): topic to check

        Returns:
            set: partition ids
        """
    return self._client.cluster.partitions_for_topic(topic)