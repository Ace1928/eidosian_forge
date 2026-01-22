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
def execute_success_callback(self, death_penalty_class: Type[BaseDeathPenalty], result: Any):
    """Executes success_callback for a job.
        with timeout .

        Args:
            death_penalty_class (Type[BaseDeathPenalty]): The penalty class to use for timeout
            result (Any): The job's result.
        """
    if not self.success_callback:
        return
    logger.debug('Running success callbacks for %s', self.id)
    with death_penalty_class(self.success_callback_timeout, JobTimeoutException, job_id=self.id):
        self.success_callback(self, self.connection, result)