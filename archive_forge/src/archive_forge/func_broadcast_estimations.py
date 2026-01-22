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
def broadcast_estimations(self, concurrency_id: str, after: int | None=None) -> None:
    wait_so_far = 0
    event_queue = self.event_queue_per_concurrency_id[concurrency_id]
    time_till_available_worker: int | None = 0
    if event_queue.current_concurrency == event_queue.concurrency_limit:
        expected_end_times = []
        for fn_index, start_times in event_queue.start_times_per_fn_index.items():
            if fn_index not in self.process_time_per_fn_index:
                time_till_available_worker = None
                break
            process_time = self.process_time_per_fn_index[fn_index].avg_time
            expected_end_times += [start_time + process_time for start_time in start_times]
        if time_till_available_worker is not None and len(expected_end_times) > 0:
            time_of_first_completion = min(expected_end_times)
            time_till_available_worker = max(time_of_first_completion - time.time(), 0)
    for rank, event in enumerate(event_queue.queue):
        process_time_for_fn = self.process_time_per_fn_index[event.fn_index].avg_time if event.fn_index in self.process_time_per_fn_index else None
        rank_eta = process_time_for_fn + wait_so_far + time_till_available_worker if process_time_for_fn is not None and wait_so_far is not None and (time_till_available_worker is not None) else None
        if after is None or rank >= after:
            self.send_message(event, EstimationMessage(rank=rank, rank_eta=rank_eta, queue_size=len(event_queue.queue)))
        if event_queue.concurrency_limit is None:
            wait_so_far = 0
        elif wait_so_far is not None and process_time_for_fn is not None:
            wait_so_far += process_time_for_fn / event_queue.concurrency_limit
        else:
            wait_so_far = None