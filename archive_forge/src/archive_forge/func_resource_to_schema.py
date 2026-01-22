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
def resource_to_schema(resource: StateResource) -> StateSchema:
    if resource == StateResource.ACTORS:
        return ActorState
    elif resource == StateResource.JOBS:
        return JobState
    elif resource == StateResource.NODES:
        return NodeState
    elif resource == StateResource.OBJECTS:
        return ObjectState
    elif resource == StateResource.PLACEMENT_GROUPS:
        return PlacementGroupState
    elif resource == StateResource.RUNTIME_ENVS:
        return RuntimeEnvState
    elif resource == StateResource.TASKS:
        return TaskState
    elif resource == StateResource.WORKERS:
        return WorkerState
    elif resource == StateResource.CLUSTER_EVENTS:
        return ClusterEventState
    else:
        assert False, 'Unreachable'