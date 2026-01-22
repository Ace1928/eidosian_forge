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
def build_binary_annotation(key, value, endpoint=None):
    annotation_type = ttypes.AnnotationType.STRING
    return ttypes.BinaryAnnotation(key, value, annotation_type, endpoint)