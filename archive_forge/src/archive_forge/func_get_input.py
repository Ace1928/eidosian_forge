import asyncio
from collections import deque, defaultdict
import dataclasses
from dataclasses import field
import logging
from typing import List, Dict, Optional, Set, Deque, Callable
import ray
from ray.workflow.common import (
from ray.workflow.workflow_context import WorkflowTaskContext
def get_input(self, task_id: TaskID) -> Optional[WorkflowRef]:
    """Get the input. It checks memory first and storage later. It returns None if
        the input does not exist.
        """
    return self.output_map.get(task_id, self.checkpoint_map.get(task_id))