from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def offsets_to_commit(self):
    if self._txn_consumer_group is None:
        return
    for group_id, offsets, _ in self._pending_txn_offsets:
        return (offsets, group_id)