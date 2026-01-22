import datetime
from itertools import count
import os
import threading
import time
import urllib.parse
import pytest
import cherrypy
from cherrypy.lib import httputil
from cherrypy.test import helper
@pytest.mark.xfail(reason='#1536')
def test_antistampede(self):
    SECONDS = 4
    slow_url = '/long_process?seconds={SECONDS}'.format(**locals())
    self.getPage(slow_url)
    self.assertBody('success!')
    path = urllib.parse.quote(slow_url, safe='')
    self.getPage('/clear_cache?path=' + path)
    self.assertStatus(200)
    start = datetime.datetime.now()

    def run():
        self.getPage(slow_url)
        self.assertBody('success!')
    ts = [threading.Thread(target=run) for i in range(100)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    finish = datetime.datetime.now()
    allowance = SECONDS + 2
    self.assertEqualDates(start, finish, seconds=allowance)