import logging
import sys
import traceback
import uuid
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import total_ordering
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from redis import WatchError
from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty
from .connections import resolve_connection
from .defaults import DEFAULT_RESULT_TTL
from .dependency import Dependency
from .exceptions import DequeueTimeout, NoSuchJobError
from .job import Callback, Job, JobStatus
from .logutils import blue, green
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import as_text, backend_class, compact, get_version, import_attribute, parse_timeout, utcnow
def enqueue_call(self, func: 'FunctionReferenceType', args: Union[Tuple, List, None]=None, kwargs: Optional[Dict]=None, timeout: Optional[int]=None, result_ttl: Optional[int]=None, ttl: Optional[int]=None, failure_ttl: Optional[int]=None, description: Optional[str]=None, depends_on: Optional['JobDependencyType']=None, job_id: Optional[str]=None, at_front: bool=False, meta: Optional[Dict]=None, retry: Optional['Retry']=None, on_success: Optional[Union[Callback, Callable[..., Any]]]=None, on_failure: Optional[Union[Callback, Callable[..., Any]]]=None, on_stopped: Optional[Union[Callback, Callable[..., Any]]]=None, pipeline: Optional['Pipeline']=None) -> Job:
    """Creates a job to represent the delayed function call and enqueues it.

        It is much like `.enqueue()`, except that it takes the function's args
        and kwargs as explicit arguments.  Any kwargs passed to this function
        contain options for RQ itself.

        Args:
            func (FunctionReferenceType): The reference to the function
            args (Union[Tuple, List, None], optional): THe `*args` to pass to the function. Defaults to None.
            kwargs (Optional[Dict], optional): THe `**kwargs` to pass to the function. Defaults to None.
            timeout (Optional[int], optional): Function timeout. Defaults to None.
            result_ttl (Optional[int], optional): Result time to live. Defaults to None.
            ttl (Optional[int], optional): Time to live. Defaults to None.
            failure_ttl (Optional[int], optional): Failure time to live. Defaults to None.
            description (Optional[str], optional): The job description. Defaults to None.
            depends_on (Optional[JobDependencyType], optional): The job dependencies. Defaults to None.
            job_id (Optional[str], optional): The job ID. Defaults to None.
            at_front (bool, optional): Whether to enqueue the job at the front. Defaults to False.
            meta (Optional[Dict], optional): Metadata to attach to the job. Defaults to None.
            retry (Optional[Retry], optional): Retry object. Defaults to None.
            on_success (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on success. Defaults to
                None. Callable is deprecated.
            on_failure (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on failure. Defaults to
                None. Callable is deprecated.
            on_stopped (Optional[Union[Callback, Callable[..., Any]]], optional): Callback for on stopped. Defaults to
                None. Callable is deprecated.
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.

        Returns:
            Job: The enqueued Job
        """
    job = self.create_job(func, args=args, kwargs=kwargs, result_ttl=result_ttl, ttl=ttl, failure_ttl=failure_ttl, description=description, depends_on=depends_on, job_id=job_id, meta=meta, status=JobStatus.QUEUED, timeout=timeout, retry=retry, on_success=on_success, on_failure=on_failure, on_stopped=on_stopped)
    return self.enqueue_job(job, pipeline=pipeline, at_front=at_front)