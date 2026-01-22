import asyncio
import datetime
import hashlib
import inspect
import re
import time as mod_time
import warnings
from typing import (
from aioredis.compat import Protocol, TypedDict
from aioredis.connection import (
from aioredis.exceptions import (
from aioredis.lock import Lock
from aioredis.utils import safe_str, str_if_bytes
def parse_georadius_generic(response, **options):
    if options['store'] or options['store_dist']:
        return response
    if type(response) != list:
        response_list = [response]
    else:
        response_list = response
    if not options['withdist'] and (not options['withcoord']) and (not options['withhash']):
        return response_list
    cast: Dict[str, Callable] = {'withdist': float, 'withcoord': lambda ll: (float(ll[0]), float(ll[1])), 'withhash': int}
    f = [lambda x: x]
    f += [cast[o] for o in ['withdist', 'withhash', 'withcoord'] if options[o]]
    return [list(map(lambda fv: fv[0](fv[1]), zip(f, r))) for r in response_list]