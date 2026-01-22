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
def test_1_Concurrency(self):
    client_thread_count = 5
    request_count = 30
    self.getPage('/')
    assert self.body == b'1'
    cookies = self.cookies
    data_dict = {}

    def request(index):
        for i in range(request_count):
            self.getPage('/', cookies)
        if not self.body.isdigit():
            self.fail(self.body)
        data_dict[index] = int(self.body)
    ts = []
    for c in range(client_thread_count):
        data_dict[c] = 0
        t = threading.Thread(target=request, args=(c,))
        ts.append(t)
        t.start()
    for t in ts:
        t.join()
    hitcount = max(data_dict.values())
    expected = 1 + client_thread_count * request_count
    assert hitcount == expected