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
def rest_response(success, message, convert_google_style=True, reason=None, **kwargs) -> aiohttp.web.Response:
    if os.environ.get('RAY_DASHBOARD_DEV') == '1':
        headers = {'Access-Control-Allow-Origin': '*'}
    else:
        headers = {}
    return aiohttp.web.json_response({'result': success, 'msg': message, 'data': to_google_style(kwargs) if convert_google_style else kwargs}, dumps=functools.partial(json.dumps, cls=CustomEncoder), headers=headers, status=200 if success else 500, reason=reason)