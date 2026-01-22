import sys
from billiard.einfo import ExceptionInfo, ExceptionWithTraceback
from kombu import serialization
from kombu.exceptions import OperationalError
from kombu.utils.uuid import uuid
from celery import current_app, states
from celery._state import _task_stack
from celery.canvas import _chain, group, signature
from celery.exceptions import Ignore, ImproperlyConfigured, MaxRetriesExceededError, Reject, Retry
from celery.local import class_property
from celery.result import EagerResult, denied_join_result
from celery.utils import abstract
from celery.utils.functional import mattrgetter, maybe_list
from celery.utils.imports import instantiate
from celery.utils.nodenames import gethostname
from celery.utils.serialization import raise_with_context
from .annotations import resolve_all as resolve_all_annotations
from .registry import _unpickle_task_v2
from .utils import appstr
def shadow_name(self, args, kwargs, options):
    """Override for custom task name in worker logs/monitoring.

        Example:
            .. code-block:: python

                from celery.utils.imports import qualname

                def shadow_name(task, args, kwargs, options):
                    return qualname(args[0])

                @app.task(shadow_name=shadow_name, serializer='pickle')
                def apply_function_async(fun, *args, **kwargs):
                    return fun(*args, **kwargs)

        Arguments:
            args (Tuple): Task positional arguments.
            kwargs (Dict): Task keyword arguments.
            options (Dict): Task execution options.
        """