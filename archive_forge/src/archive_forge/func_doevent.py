import logging
import sys
import types
import threading
import inspect
from functools import wraps
from itertools import chain
from numba.core import config
def doevent(msg):
    msg = ['== ', tls.indent * ' ', msg]
    logger = logging.getLogger('trace')
    logger.info(''.join(msg))