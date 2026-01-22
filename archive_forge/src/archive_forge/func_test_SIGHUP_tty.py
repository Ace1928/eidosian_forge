import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def test_SIGHUP_tty(self):
    try:
        from signal import SIGHUP
    except ImportError:
        return self.skip('skipped (no SIGHUP) ')
    p = helper.CPProcess(ssl=self.scheme.lower() == 'https')
    p.write_conf(extra='test_case_name: "test_SIGHUP_tty"')
    p.start(imports='cherrypy.test._test_states_demo')
    os.kill(p.get_pid(), SIGHUP)
    p.join()