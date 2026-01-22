import socket
import sys
import time
import eventlet.wsgi
import greenlet
from oslo_config import cfg
from oslo_service import service
def hi_app(environ, start_response):
    time.sleep(process_time)
    start_response('200 OK', [('Content-Type', 'application/json')])
    yield 'hi'