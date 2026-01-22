from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def complete_transaction(self):
    assert not self._pending_txn_partitions
    assert not self._pending_txn_offsets
    self._transition_to(TransactionState.READY)
    self._txn_partitions.clear()
    self._txn_consumer_group = None
    if not self._transaction_waiter.done():
        self._transaction_waiter.set_result(None)