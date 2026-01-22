import asyncio
from collections import deque, defaultdict
import dataclasses
from dataclasses import field
import logging
from typing import List, Dict, Optional, Set, Deque, Callable
import ray
from ray.workflow.common import (
from ray.workflow.workflow_context import WorkflowTaskContext
def pop_frontier_to_run(self) -> Optional[TaskID]:
    """Pop one task to run from the frontier queue."""
    try:
        t = self.frontier_to_run.popleft()
        self.frontier_to_run_set.remove(t)
        return t
    except IndexError:
        return None