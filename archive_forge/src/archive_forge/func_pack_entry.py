import json
import uuid
from redis import asyncio as aioredis
from . import defaults
from .base_cache import BaseCache, CacheEntry
def pack_entry(entry):
    ts, pol_id, pol_body = entry
    obj = (pol_id, pol_body)
    packed = uuid.uuid4().bytes + json.dumps(obj).encode('utf-8')
    return packed