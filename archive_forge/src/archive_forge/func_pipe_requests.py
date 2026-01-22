import collections
import threading
import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import abandonment
from grpc.framework.foundation import logging_pool
from grpc.framework.foundation import stream
from grpc.framework.interfaces.face import face
def pipe_requests():
    for request in request_iterator:
        if not servicer_context.is_active() or thread_joined.is_set():
            return
        request_consumer.consume(request)
        if not servicer_context.is_active() or thread_joined.is_set():
            return
    request_consumer.terminate()