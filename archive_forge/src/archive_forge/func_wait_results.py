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
def wait_results(self, result, response):
    result.get(propagate=False)
    self.update_response_result(response, result)
    if self.backend_configured(result):
        response.update(state=result.state)
    return response