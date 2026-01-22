import datetime
import json
import logging
import sys
from abc import ABC
from dataclasses import asdict, field, fields
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import ray.dashboard.utils as dashboard_utils
from ray._private.ray_constants import env_integer
from ray.core.generated.common_pb2 import TaskStatus, TaskType
from ray.core.generated.gcs_pb2 import TaskEvents
from ray.util.state.custom_types import (
from ray.util.state.exception import RayStateApiException
from ray.dashboard.modules.job.pydantic_models import JobDetails
from ray._private.pydantic_compat import IS_PYDANTIC_2
def get_pending_tasks_count(task_group: NestedTaskSummary) -> int:
    return task_group.state_counts.get('PENDING_ARGS_AVAIL', 0) + task_group.state_counts.get('PENDING_NODE_ASSIGNMENT', 0) + task_group.state_counts.get('PENDING_OBJ_STORE_MEM_AVAIL', 0) + task_group.state_counts.get('PENDING_ARGS_FETCH', 0)