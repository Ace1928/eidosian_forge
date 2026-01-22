import ray
from ray.dag.base import DAGNodeBase
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.util.annotations import DeveloperAPI
from typing import (
import uuid
import asyncio
def get_stable_uuid(self) -> str:
    """Return stable uuid for this node.
        1) Generated only once at first instance creation
        2) Stable across pickling, replacement and JSON serialization.
        """
    return self._stable_uuid