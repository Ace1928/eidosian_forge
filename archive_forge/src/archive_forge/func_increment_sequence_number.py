from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def increment_sequence_number(self, tp: TopicPartition, increment: int):
    seq = self._sequence_numbers[tp] + increment
    if seq > 2 ** 31 - 1:
        seq -= 2 ** 32
    self._sequence_numbers[tp] = seq