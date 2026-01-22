import os
import sys
import time
import struct
import socket
import random
from eventlet.green import threading
from eventlet.zipkin._thrift.zipkinCore import ttypes
from eventlet.zipkin._thrift.zipkinCore.constants import SERVER_SEND
@staticmethod
def build_span(name, trace_id, span_id, parent_id, annotations, bannotations):
    return ttypes.Span(name=name, trace_id=trace_id, id=span_id, parent_id=parent_id, annotations=annotations, binary_annotations=bannotations)