import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def get_group_by_type(group_by: str):
    """Translate string input into GroupByType instance"""
    group_by = group_by.upper()
    if group_by == 'NODE_ADDRESS':
        return GroupByType.NODE_ADDRESS
    elif group_by == 'STACK_TRACE':
        return GroupByType.STACK_TRACE
    else:
        raise Exception('The group-by input provided is not one of                NODE_ADDRESS or STACK_TRACE.')