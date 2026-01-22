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
def get_or_create_task_group(task_id: str) -> Optional[NestedTaskSummary]:
    """
            Gets an already created task_group
            OR
            Creates a task group and puts it in the right place under its parent.
            For actor tasks, the parent is the Actor that owns it. For all other
            tasks, the owner is the driver or task that created it.

            Returns None if there is missing data about the task or one of its parents.

            For task groups that represents actors, the id is in the
            format actor:{actor_id}
            """
    if task_id in task_group_by_id:
        return task_group_by_id[task_id]
    task = tasks_by_id.get(task_id)
    if not task:
        logger.debug(f"We're missing data about {task_id}")
        return None
    func_name = task['name'] or task['func_or_class_name']
    task_id = task['task_id']
    type_enum = TaskType.DESCRIPTOR.values_by_name[task['type']].number
    task_group_by_id[task_id] = NestedTaskSummary(name=func_name, key=task_id, type=task['type'], timestamp=task['creation_time_ms'], link=Link(type='task', id=task_id))
    if type_enum == TaskType.ACTOR_TASK or type_enum == TaskType.ACTOR_CREATION_TASK:
        parent_task_group = get_or_create_actor_task_group(task['actor_id'])
        if parent_task_group:
            parent_task_group.children.append(task_group_by_id[task_id])
    else:
        parent_task_id = task['parent_task_id']
        if not parent_task_id or parent_task_id.startswith(DRIVER_TASK_ID_PREFIX):
            summary.append(task_group_by_id[task_id])
        else:
            parent_task_group = get_or_create_task_group(parent_task_id)
            if parent_task_group:
                parent_task_group.children.append(task_group_by_id[task_id])
    return task_group_by_id[task_id]