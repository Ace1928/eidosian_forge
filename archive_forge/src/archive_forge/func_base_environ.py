import io
import logging
import os
import re
import sys
from gunicorn.http.message import HEADER_RE
from gunicorn.http.errors import InvalidHeader, InvalidHeaderName
from gunicorn import SERVER_SOFTWARE, SERVER
from gunicorn import util
def base_environ(cfg):
    return {'wsgi.errors': WSGIErrorsWrapper(cfg), 'wsgi.version': (1, 0), 'wsgi.multithread': False, 'wsgi.multiprocess': cfg.workers > 1, 'wsgi.run_once': False, 'wsgi.file_wrapper': FileWrapper, 'wsgi.input_terminated': True, 'SERVER_SOFTWARE': SERVER_SOFTWARE}