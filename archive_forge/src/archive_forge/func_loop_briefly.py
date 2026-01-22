import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
def loop_briefly(self):
    """Run all queued callbacks on the IOLoop.

        In these tests, this method is used after calling notify() to
        preserve the pre-5.0 behavior in which callbacks ran
        synchronously.
        """
    self.io_loop.add_callback(self.stop)
    self.wait()