import io
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from random import randint
from ssl import SSLError
from gunicorn import util
from gunicorn.http.errors import (
from gunicorn.http.wsgi import Response, default_environ
from gunicorn.reloader import reloader_engines
from gunicorn.workers.workertmp import WorkerTmp
def load_wsgi(self):
    try:
        self.wsgi = self.app.wsgi()
    except SyntaxError as e:
        if not self.cfg.reload:
            raise
        self.log.exception(e)
        try:
            _, exc_val, exc_tb = sys.exc_info()
            self.reloader.add_extra_file(exc_val.filename)
            tb_string = io.StringIO()
            traceback.print_tb(exc_tb, file=tb_string)
            self.wsgi = util.make_fail_app(tb_string.getvalue())
        finally:
            del exc_tb