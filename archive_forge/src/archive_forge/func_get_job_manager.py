import aiohttp
from aiohttp.web import Request, Response
import dataclasses
import json
import logging
import traceback
import ray
import ray.dashboard.optional_utils as optional_utils
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.job.job_manager import JobManager
from ray.dashboard.modules.job.pydantic_models import JobType
from ray.dashboard.modules.job.utils import parse_and_validate_request, find_job_by_ids
def get_job_manager(self):
    if not self._job_manager:
        self._job_manager = JobManager(self._dashboard_agent.gcs_aio_client, self._dashboard_agent.log_dir)
    return self._job_manager