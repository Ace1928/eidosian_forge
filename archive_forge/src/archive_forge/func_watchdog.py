import os
import sys
import tornado.web
import tornado.httpserver
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.wsgi import WSGIContainer
from gunicorn.workers.base import Worker
from gunicorn import __version__ as gversion
from gunicorn.sock import ssl_context
def watchdog(self):
    if self.alive:
        self.notify()
    if self.ppid != os.getppid():
        self.log.info('Parent changed, shutting down: %s', self)
        self.alive = False