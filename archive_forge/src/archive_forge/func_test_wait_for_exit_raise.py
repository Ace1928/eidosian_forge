import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
import unittest
from tornado.httpclient import HTTPClient, HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.log import gen_log
from tornado.process import fork_processes, task_id, Subprocess
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.testing import bind_unused_port, ExpectLog, AsyncTestCase, gen_test
from tornado.test.util import skipIfNonUnix
from tornado.web import RequestHandler, Application
@gen_test
def test_wait_for_exit_raise(self):
    Subprocess.initialize()
    self.addCleanup(Subprocess.uninitialize)
    subproc = Subprocess([sys.executable, '-c', 'import sys; sys.exit(1)'])
    with self.assertRaises(subprocess.CalledProcessError) as cm:
        yield subproc.wait_for_exit()
    self.assertEqual(cm.exception.returncode, 1)