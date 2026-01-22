import collections
import copy
import logging
import threading
import time
from concurrent.futures import Future
from aiokafka import errors as Errors
from aiokafka.conn import collect_hosts
from aiokafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition
def brokers(self):
    """Get all BrokerMetadata

        Returns:
            set: {BrokerMetadata, ...}
        """
    return set(self._brokers.values()) or set(self._bootstrap_brokers.values())