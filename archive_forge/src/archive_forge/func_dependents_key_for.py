import asyncio
import inspect
import json
import logging
import warnings
import zlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from uuid import uuid4
from redis import WatchError
from .defaults import CALLBACK_TIMEOUT, UNSERIALIZABLE_RETURN_VALUE_PAYLOAD
from .timeouts import BaseDeathPenalty, JobTimeoutException
from .connections import resolve_connection
from .exceptions import DeserializationError, InvalidJobOperation, NoSuchJobError
from .local import LocalStack
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import (
@classmethod
def dependents_key_for(cls, job_id: str) -> str:
    """The Redis key that is used to store job dependents hash under.

        Args:
            job_id (str): The "parent" job id

        Returns:
            dependents_key (str): The dependents key
        """
    return '{0}{1}:dependents'.format(cls.redis_job_namespace_prefix, job_id)