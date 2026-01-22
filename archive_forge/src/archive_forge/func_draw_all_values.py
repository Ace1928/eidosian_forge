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
def draw_all_values(self):
    with self._condition:
        while True:
            if self._cancelled:
                raise abandonment.Abandoned()
            elif self._terminated:
                all_values = tuple(self._values)
                self._values = None
                return all_values
            else:
                self._condition.wait()