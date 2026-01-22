import json
import asyncio
import logging
import dataclasses
from functools import wraps
from typing import Union
import aiohttp
from aiohttp.web import Request, Response
import ray
from ray.exceptions import RayTaskError
from ray.dashboard.modules.version import (
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as optional_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray._private.pydantic_compat import ValidationError
def validate_http_options(self, client, http_options):
    divergent_http_options = []
    for option, new_value in http_options.items():
        prev_value = getattr(client.http_config, option)
        if prev_value != new_value:
            divergent_http_options.append(option)
    if divergent_http_options:
        logger.warning(f"Serve is already running on this Ray cluster and it's not possible to update its HTTP options without restarting it. Following options are attempted to be updated: {divergent_http_options}.")