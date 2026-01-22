import asyncio
import logging
import weakref
from ._asyncio_loop import get_running_loop, get_task_loop
def tasks_to_complete(self):
    return []