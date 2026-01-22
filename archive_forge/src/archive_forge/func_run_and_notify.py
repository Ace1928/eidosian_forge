import asyncio
import concurrent.futures
import logging
import queue
import sys
import threading
from typing import (
from wandb.errors.term import termerror
from wandb.filesync import upload_job
from wandb.sdk.lib.paths import LogicalPath
def run_and_notify() -> None:
    try:
        self._do_upload_sync(event)
    finally:
        self._event_queue.put(EventJobDone(event, exc=sys.exc_info()[1]))