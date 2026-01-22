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
def build_endpoint(ipv4=None, port=None, service_name=None):
    if ipv4 is not None:
        ipv4 = ZipkinDataBuilder._ipv4_to_int(ipv4)
    if service_name is None:
        service_name = ZipkinDataBuilder._get_script_name()
    return ttypes.Endpoint(ipv4=ipv4, port=port, service_name=service_name)