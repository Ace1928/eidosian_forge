from datetime import datetime
import errno
import socket
import ssl
import sys
from gunicorn import http
from gunicorn.http import wsgi
from gunicorn import util
from gunicorn.workers import base
def is_already_handled(self, respiter):
    return respiter == ALREADY_HANDLED