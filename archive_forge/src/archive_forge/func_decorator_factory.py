import asyncio
import collections
import functools
import inspect
import json
import logging
import os
import time
import traceback
from collections import namedtuple
from typing import Any, Callable
from aiohttp.web import Response
import ray
import ray.dashboard.consts as dashboard_consts
from ray._private.ray_constants import RAY_INTERNAL_DASHBOARD_NAMESPACE, env_bool
from ray.dashboard.optional_deps import PathLike, RouteDef, aiohttp, hdrs
from ray.dashboard.utils import CustomEncoder, to_google_style
def decorator_factory(f: Callable) -> Callable:

    @functools.wraps(f)
    async def decorator(self, *args, **kwargs):
        try:
            if not ray.is_initialized():
                try:
                    address = self.get_gcs_address()
                    logger.info(f'Connecting to ray with address={address}')
                    os.environ['RAY_gcs_server_request_timeout_seconds'] = str(dashboard_consts.GCS_RPC_TIMEOUT_SECONDS)
                    ray.init(address=address, log_to_driver=False, configure_logging=False, namespace=RAY_INTERNAL_DASHBOARD_NAMESPACE, _skip_env_hook=True)
                except Exception as e:
                    ray.shutdown()
                    raise e from None
            return await f(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f'Unexpected error in handler: {e}')
            return Response(text=traceback.format_exc(), status=aiohttp.web.HTTPInternalServerError.status_code)
    return decorator