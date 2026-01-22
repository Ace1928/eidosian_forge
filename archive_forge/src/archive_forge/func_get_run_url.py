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
def get_run_url(self) -> str:
    """Get the LangSmith root run URL"""
    if not self.latest_run:
        raise ValueError('No traced run found.')
    for attempt in Retrying(stop=stop_after_attempt(5), wait=wait_exponential_jitter(), retry=retry_if_exception_type(ls_utils.LangSmithError)):
        with attempt:
            return self.client.get_run_url(run=self.latest_run, project_name=self.project_name)
    raise ValueError('Failed to get run URL.')