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
def test_multi_process(self):
    with ExpectLog(gen_log, '(Starting .* processes|child .* exited|uncaught exception)'):
        sock, port = bind_unused_port()

        def get_url(path):
            return 'http://127.0.0.1:%d%s' % (port, path)
        signal.alarm(5)
        try:
            id = fork_processes(3, max_restarts=3)
            self.assertTrue(id is not None)
            signal.alarm(5)
        except SystemExit as e:
            self.assertEqual(e.code, 0)
            self.assertTrue(task_id() is None)
            sock.close()
            return
        try:
            if id in (0, 1):
                self.assertEqual(id, task_id())

                async def f():
                    server = HTTPServer(self.get_app())
                    server.add_sockets([sock])
                    await asyncio.Event().wait()
                asyncio.run(f())
            elif id == 2:
                self.assertEqual(id, task_id())
                sock.close()
                client = HTTPClient(SimpleAsyncHTTPClient)

                def fetch(url, fail_ok=False):
                    try:
                        return client.fetch(get_url(url))
                    except HTTPError as e:
                        if not (fail_ok and e.code == 599):
                            raise
                fetch('/?exit=2', fail_ok=True)
                fetch('/?exit=3', fail_ok=True)
                int(fetch('/').body)
                fetch('/?exit=0', fail_ok=True)
                pid = int(fetch('/').body)
                fetch('/?exit=4', fail_ok=True)
                pid2 = int(fetch('/').body)
                self.assertNotEqual(pid, pid2)
                fetch('/?exit=0', fail_ok=True)
                os._exit(0)
        except Exception:
            logging.error('exception in child process %d', id, exc_info=True)
            raise