import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
def test_signal_handler_unsubscribe(self):
    self._require_signal_and_kill('SIGTERM')
    if os.name == 'nt':
        self.skip('SIGTERM not available')
    p = helper.CPProcess(ssl=self.scheme.lower() == 'https')
    p.write_conf(extra='unsubsig: True\ntest_case_name: "test_signal_handler_unsubscribe"\n')
    p.start(imports='cherrypy.test._test_states_demo')
    os.kill(p.get_pid(), signal.SIGTERM)
    p.join()
    with open(p.error_log, 'rb') as f:
        log_lines = list(f)
        assert any((line.endswith(b'I am an old SIGTERM handler.\n') for line in log_lines))