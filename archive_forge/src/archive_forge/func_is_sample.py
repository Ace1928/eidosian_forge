import os
import sys
import time
import struct
import socket
import random
from eventlet.green import threading
from eventlet.zipkin._thrift.zipkinCore import ttypes
from eventlet.zipkin._thrift.zipkinCore.constants import SERVER_SEND
def is_sample():
    """ Return whether it should record trace information
        for the request or not
    """
    return is_tracing() and _tls.trace_data.sampled