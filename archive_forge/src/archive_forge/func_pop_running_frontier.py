import asyncio
from collections import deque, defaultdict
import dataclasses
from dataclasses import field
import logging
from typing import List, Dict, Optional, Set, Deque, Callable
import ray
from ray.workflow.common import (
from ray.workflow.workflow_context import WorkflowTaskContext
def pop_running_frontier(self, fut: asyncio.Future) -> WorkflowRef:
    """Pop a task from the running frontier."""
    ref = self.running_frontier.pop(fut)
    self.running_frontier_set.remove(ref.task_id)
    return ref