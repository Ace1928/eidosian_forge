import json
import uuid
from redis import asyncio as aioredis
from . import defaults
from .base_cache import BaseCache, CacheEntry
def unpack_entry(packed):
    bin_obj = packed[16:]
    obj = json.loads(bin_obj.decode('utf-8'))
    pol_id, pol_body = obj
    return CacheEntry(ts=0, pol_id=pol_id, pol_body=pol_body)