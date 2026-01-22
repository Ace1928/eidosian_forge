import sys
import logging
from concurrent.futures import ThreadPoolExecutor
import celery
import tornado.web
from tornado import ioloop
from tornado.httpserver import HTTPServer
from tornado.web import url
from .urls import handlers as default_handlers
from .events import Events
from .inspector import Inspector
from .options import default_options
def update_workers(self, workername=None):
    return self.inspector.inspect(workername)