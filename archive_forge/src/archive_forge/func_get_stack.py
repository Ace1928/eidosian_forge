import os
import sys
from .. import config, tests, trace
from ..transport.http import opt_ssl_ca_certs, ssl
def get_stack(self, content):
    return config.MemoryStack(content.encode('utf-8'))