from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def is_empty_transaction(self):
    return len(self.txn_partitions) == 0 and self._txn_consumer_group is None