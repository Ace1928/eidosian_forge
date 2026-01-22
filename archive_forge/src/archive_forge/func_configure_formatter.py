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
def configure_formatter(self, config):
    """Configure a formatter from a dictionary."""
    if '()' in config:
        factory = config['()']
        try:
            result = self.configure_custom(config)
        except TypeError as te:
            if "'format'" not in str(te):
                raise
            config['fmt'] = config.pop('format')
            config['()'] = factory
            result = self.configure_custom(config)
    else:
        fmt = config.get('format', None)
        dfmt = config.get('datefmt', None)
        style = config.get('style', '%')
        cname = config.get('class', None)
        if not cname:
            c = logging.Formatter
        else:
            c = _resolve(cname)
        if 'validate' in config:
            result = c(fmt, dfmt, style, config['validate'])
        else:
            result = c(fmt, dfmt, style)
    return result