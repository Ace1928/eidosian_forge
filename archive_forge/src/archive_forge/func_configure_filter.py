import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
def configure_filter(self, config):
    """Configure a filter from a dictionary."""
    if '()' in config:
        result = self.configure_custom(config)
    else:
        name = config.get('name', '')
        result = logging.Filter(name)
    return result