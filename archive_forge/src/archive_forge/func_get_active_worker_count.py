from __future__ import annotations
import asyncio
import copy
import os
import random
import time
import traceback
import uuid
from collections import defaultdict
from queue import Queue as ThreadQueue
from typing import TYPE_CHECKING
import fastapi
from typing_extensions import Literal
from gradio import route_utils, routes
from gradio.data_classes import (
from gradio.exceptions import Error
from gradio.helpers import TrackedIterable
from gradio.server_messages import (
from gradio.utils import LRUCache, run_coro_in_background, safe_get_lock, set_task_name
def get_active_worker_count(self) -> int:
    count = 0
    for worker in self.active_jobs:
        if worker is not None:
            count += 1
    return count