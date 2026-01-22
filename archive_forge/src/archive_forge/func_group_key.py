import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def group_key(self, group_by_type: GroupByType) -> str:
    if group_by_type == GroupByType.NODE_ADDRESS:
        return self.node_address
    elif group_by_type == GroupByType.STACK_TRACE:
        return self.call_site
    else:
        raise ValueError(f'group by type {group_by_type} is invalid.')