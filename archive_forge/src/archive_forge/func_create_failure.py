import zlib
from base64 import b64decode, b64encode
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from redis import Redis
from .defaults import UNSERIALIZABLE_RETURN_VALUE_PAYLOAD
from .job import Job
from .serializers import resolve_serializer
from .utils import decode_redis_hash, now
@classmethod
def create_failure(cls, job, ttl, exc_string, pipeline=None):
    result = cls(job_id=job.id, type=cls.Type.FAILED, connection=job.connection, exc_string=exc_string, serializer=job.serializer)
    result.save(ttl=ttl, pipeline=pipeline)
    return result