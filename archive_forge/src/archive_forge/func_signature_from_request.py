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
def signature_from_request(self, request=None, args=None, kwargs=None, queue=None, **extra_options):
    request = self.request if request is None else request
    args = request.args if args is None else args
    kwargs = request.kwargs if kwargs is None else kwargs
    options = {**request.as_execution_options(), **extra_options}
    delivery_info = request.delivery_info or {}
    priority = delivery_info.get('priority')
    if priority is not None:
        options['priority'] = priority
    if queue:
        options['queue'] = queue
    else:
        exchange = delivery_info.get('exchange')
        routing_key = delivery_info.get('routing_key')
        if exchange == '' and routing_key:
            options['queue'] = routing_key
        else:
            options.update(delivery_info)
    return self.signature(args, kwargs, options, type=self, **extra_options)