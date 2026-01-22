import os
import time
import mmap
import json
import fnmatch
import asyncio
import itertools
import collections
import logging.handlers
from ray._private.utils import get_or_create_event_loop
from concurrent.futures import ThreadPoolExecutor
from ray._private.utils import run_background_task
from ray.dashboard.modules.event import event_consts
from ray.dashboard.utils import async_loop_forever
def parse_event_strings(event_string_list):
    events = []
    for data in event_string_list:
        if not data:
            continue
        try:
            event = _parse_line(data)
            events.append(event)
        except Exception:
            logger.exception('Parse event line failed: %s', repr(data))
    return events