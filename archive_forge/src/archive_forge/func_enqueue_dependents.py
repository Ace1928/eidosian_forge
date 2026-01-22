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
def enqueue_dependents(self, job: 'Job', pipeline: Optional['Pipeline']=None, exclude_job_id: Optional[str]=None):
    """Enqueues all jobs in the given job's dependents set and clears it.

        When called without a pipeline, this method uses WATCH/MULTI/EXEC.
        If you pass a pipeline, only MULTI is called. The rest is up to the
        caller.

        Args:
            job (Job): The Job to enqueue the dependents
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.
            exclude_job_id (Optional[str], optional): Whether to exclude the job id. Defaults to None.
        """
    from .registry import DeferredJobRegistry
    pipe = pipeline if pipeline is not None else self.connection.pipeline()
    dependents_key = job.dependents_key
    while True:
        try:
            if pipeline is None:
                pipe.watch(dependents_key)
            dependent_job_ids = {as_text(_id) for _id in pipe.smembers(dependents_key)}
            if not dependent_job_ids:
                break
            jobs_to_enqueue = [dependent_job for dependent_job in self.job_class.fetch_many(dependent_job_ids, connection=self.connection, serializer=self.serializer) if dependent_job and dependent_job.dependencies_are_met(parent_job=job, pipeline=pipe, exclude_job_id=exclude_job_id) and (dependent_job.get_status(refresh=False) != JobStatus.CANCELED)]
            pipe.multi()
            if not jobs_to_enqueue:
                break
            for dependent in jobs_to_enqueue:
                enqueue_at_front = dependent.enqueue_at_front or False
                registry = DeferredJobRegistry(dependent.origin, self.connection, job_class=self.job_class, serializer=self.serializer)
                registry.remove(dependent, pipeline=pipe)
                if dependent.origin == self.name:
                    self._enqueue_job(dependent, pipeline=pipe, at_front=enqueue_at_front)
                else:
                    queue = self.__class__(name=dependent.origin, connection=self.connection)
                    queue._enqueue_job(dependent, pipeline=pipe, at_front=enqueue_at_front)
            if len(jobs_to_enqueue) == len(dependent_job_ids):
                pipe.delete(dependents_key)
            else:
                enqueued_job_ids = [job.id for job in jobs_to_enqueue]
                pipe.srem(dependents_key, *enqueued_job_ids)
            if pipeline is None:
                pipe.execute()
            break
        except WatchError:
            if pipeline is None:
                continue
            else:
                raise