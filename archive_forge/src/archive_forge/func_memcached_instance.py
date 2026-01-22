import os
import platform
import threading
import time
from http.client import HTTPConnection
from distutils.spawn import find_executable
import pytest
from path import Path
from more_itertools import consume
import portend
import cherrypy
from cherrypy._cpcompat import HTTPSConnection
from cherrypy.lib import sessions
from cherrypy.lib import reprconf
from cherrypy.lib.httputil import response_codes
from cherrypy.test import helper
from cherrypy import _json as json
@pytest.fixture(scope='session')
def memcached_instance(request, watcher_getter, memcached_server_present):
    """
    Start up an instance of memcached.
    """
    port = portend.find_available_local_port()

    def is_occupied():
        try:
            portend.Checker().assert_free('localhost', port)
        except Exception:
            return True
        return False
    proc = watcher_getter(name='memcached', arguments=['-p', str(port)], checker=is_occupied, request=request)
    return locals()