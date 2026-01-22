import asyncio
from datetime import timedelta
import typing  # noqa: F401
import unittest
from tornado import gen, locks
from tornado.gen import TimeoutError
from tornado.testing import gen_test, AsyncTestCase
@gen_test
def test_context_manager_contended(self):
    sem = locks.Semaphore()
    history = []

    @gen.coroutine
    def f(index):
        with (yield sem.acquire()):
            history.append('acquired %d' % index)
            yield gen.sleep(0.01)
            history.append('release %d' % index)
    yield [f(i) for i in range(2)]
    expected_history = []
    for i in range(2):
        expected_history.extend(['acquired %d' % i, 'release %d' % i])
    self.assertEqual(expected_history, history)