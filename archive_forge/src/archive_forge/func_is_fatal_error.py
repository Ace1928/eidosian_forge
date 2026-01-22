from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def is_fatal_error(self):
    return self.state == TransactionState.FATAL_ERROR