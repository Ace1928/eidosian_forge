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
def configure_root(self, config, incremental=False):
    """Configure a root logger from a dictionary."""
    root = logging.getLogger()
    self.common_logger_config(root, config, incremental)