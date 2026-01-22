from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import UUID
from langsmith import Client
from langsmith import utils as ls_utils
from tenacity import (
from langchain_core.env import get_runtime_environment
from langchain_core.load import dumpd
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
def wait_for_all_tracers() -> None:
    """Wait for all tracers to finish."""
    global _CLIENT
    if _CLIENT is not None and _CLIENT.tracing_queue is not None:
        _CLIENT.tracing_queue.join()