import json
import logging
from collections import OrderedDict
from datetime import datetime
from celery import states
from celery.backends.base import DisabledBackend
from celery.contrib.abortable import AbortableAsyncResult
from celery.result import AsyncResult
from tornado import web
from tornado.escape import json_decode
from tornado.ioloop import IOLoop
from tornado.web import HTTPError
from ..utils import tasks
from ..utils.broker import Broker
from . import BaseApiHandler
def get_task_args(self):
    try:
        body = self.request.body
        options = json_decode(body) if body else {}
    except ValueError as e:
        raise HTTPError(400, str(e)) from e
    if not isinstance(options, dict):
        raise HTTPError(400, 'invalid options')
    args = options.pop('args', [])
    kwargs = options.pop('kwargs', {})
    if not isinstance(args, (list, tuple)):
        raise HTTPError(400, 'args must be an array')
    return (args, kwargs, options)