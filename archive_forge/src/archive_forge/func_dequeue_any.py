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
@classmethod
def dequeue_any(cls, queues: List['Queue'], timeout: Optional[int], connection: Optional['Redis']=None, job_class: Optional['Job']=None, serializer: Any=None, death_penalty_class: Optional[Type[BaseDeathPenalty]]=None) -> Tuple['Job', 'Queue']:
    """Class method returning the job_class instance at the front of the given
        set of Queues, where the order of the queues is important.

        When all of the Queues are empty, depending on the `timeout` argument,
        either blocks execution of this function for the duration of the
        timeout or until new messages arrive on any of the queues, or returns
        None.

        See the documentation of cls.lpop for the interpretation of timeout.

        Args:
            queues (List[Queue]): List of queue objects
            timeout (Optional[int]): Timeout for the LPOP
            connection (Optional[Redis], optional): Redis Connection. Defaults to None.
            job_class (Optional[Type[Job]], optional): The job class. Defaults to None.
            serializer (Any, optional): Serializer to use. Defaults to None.
            death_penalty_class (Optional[Type[BaseDeathPenalty]], optional): The death penalty class. Defaults to None.

        Raises:
            e: Any exception

        Returns:
            job, queue (Tuple[Job, Queue]): A tuple of Job, Queue
        """
    job_class: Job = backend_class(cls, 'job_class', override=job_class)
    while True:
        queue_keys = [q.key for q in queues]
        if len(queue_keys) == 1 and get_version(connection) >= (6, 2, 0):
            result = cls.lmove(connection, queue_keys[0], timeout)
        else:
            result = cls.lpop(queue_keys, timeout, connection=connection)
        if result is None:
            return None
        queue_key, job_id = map(as_text, result)
        queue = cls.from_queue_key(queue_key, connection=connection, job_class=job_class, serializer=serializer, death_penalty_class=death_penalty_class)
        try:
            job = job_class.fetch(job_id, connection=connection, serializer=serializer)
        except NoSuchJobError:
            continue
        except Exception as e:
            e.job_id = job_id
            e.queue = queue
            raise e
        return (job, queue)
    return (None, None)