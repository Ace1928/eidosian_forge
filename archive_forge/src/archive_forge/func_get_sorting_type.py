import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def get_sorting_type(sort_by: str):
    """Translate string input into SortingType instance"""
    sort_by = sort_by.upper()
    if sort_by == 'PID':
        return SortingType.PID
    elif sort_by == 'OBJECT_SIZE':
        return SortingType.OBJECT_SIZE
    elif sort_by == 'REFERENCE_TYPE':
        return SortingType.REFERENCE_TYPE
    else:
        raise Exception('The sort-by input provided is not one of                PID, OBJECT_SIZE, or REFERENCE_TYPE.')