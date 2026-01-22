from typing import Dict, List, Optional
from dataclasses import dataclass
import ray
from ray import SCRIPT_MODE, LOCAL_MODE
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
def has_resources_ready(self, resource_request: ResourceRequest) -> bool:
    if resource_request not in self._requested_resources:
        return False
    available_resources = self._available_resources
    all_resources = resource_request.required_resources
    for k, v in all_resources.items():
        if available_resources.get(k, 0.0) < v:
            return False
    return True