from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def consumer_group_to_add(self):
    if self._txn_consumer_group is not None:
        return
    for group_id, _, _ in self._pending_txn_offsets:
        return group_id