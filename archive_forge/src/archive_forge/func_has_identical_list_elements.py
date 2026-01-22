import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
from aiokafka.coordinator.assignors.abstract import AbstractPartitionAssignor
from aiokafka.coordinator.assignors.sticky.partition_movements import PartitionMovements
from aiokafka.coordinator.assignors.sticky.sorted_set import SortedSet
from aiokafka.coordinator.protocol import (
from aiokafka.coordinator.protocol import Schema
from aiokafka.protocol.struct import Struct
from aiokafka.protocol.types import String, Array, Int32
from aiokafka.structs import TopicPartition
def has_identical_list_elements(list_):
    """Checks if all lists in the collection have the same members

    Arguments:
      list_: collection of lists

    Returns:
      true if all lists in the collection have the same members; false otherwise
    """
    if not list_:
        return True
    for i in range(1, len(list_)):
        if list_[i] != list_[i - 1]:
            return False
    return True