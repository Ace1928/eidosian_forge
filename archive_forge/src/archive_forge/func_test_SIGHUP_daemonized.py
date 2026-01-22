import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def test_SIGHUP_daemonized(self):
    try:
        from signal import SIGHUP
    except ImportError:
        return self.skip('skipped (no SIGHUP) ')
    if os.name not in ['posix']:
        return self.skip('skipped (not on posix) ')
    p = helper.CPProcess(ssl=self.scheme.lower() == 'https', wait=True, daemonize=True)
    p.write_conf(extra='test_case_name: "test_SIGHUP_daemonized"')
    p.start(imports='cherrypy.test._test_states_demo')
    pid = p.get_pid()
    try:
        os.kill(pid, SIGHUP)
        time.sleep(2)
        self.getPage('/pid')
        self.assertStatus(200)
        new_pid = int(self.body)
        self.assertNotEqual(new_pid, pid)
    finally:
        self.getPage('/exit')
    p.join()