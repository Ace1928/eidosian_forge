import copy
import json
import logging
import os
from typing import Optional, Tuple
import ray
from ray.serve._private.common import ServeComponentType
from ray.serve._private.constants import (
from ray.serve.schema import EncodingType, LoggingConfig
def record_factory(*args, **kwargs):
    request_context = ray.serve.context._serve_request_context.get()
    record = factory(*args, **kwargs)
    if request_context.route:
        setattr(record, SERVE_LOG_ROUTE, request_context.route)
    if request_context.request_id:
        setattr(record, SERVE_LOG_REQUEST_ID, request_context.request_id)
    if request_context.app_name:
        setattr(record, SERVE_LOG_APPLICATION, request_context.app_name)
    return record