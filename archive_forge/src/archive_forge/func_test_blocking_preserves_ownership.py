import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def test_blocking_preserves_ownership(self):
    pthread1_event = eventlet.patcher.original('threading').Event()
    pthread2_event1 = eventlet.patcher.original('threading').Event()
    pthread2_event2 = eventlet.patcher.original('threading').Event()
    thread_id = []
    owner = []

    def pthread1():
        thread_id.append(id(eventlet.greenthread.getcurrent()))
        self.mutex.acquire()
        owner.append(self.mutex.owner)
        pthread2_event1.set()
        orig_os_write = pipe_mutex.os.write

        def patched_os_write(*a, **kw):
            try:
                return orig_os_write(*a, **kw)
            finally:
                pthread1_event.wait()
        with mock.patch.object(pipe_mutex.os, 'write', patched_os_write):
            self.mutex.release()
        pthread2_event2.set()

    def pthread2():
        pthread2_event1.wait()
        thread_id.append(id(eventlet.greenthread.getcurrent()))
        self.mutex.acquire()
        pthread1_event.set()
        pthread2_event2.wait()
        owner.append(self.mutex.owner)
        self.mutex.release()
    real_thread1 = eventlet.patcher.original('threading').Thread(target=pthread1)
    real_thread1.start()
    real_thread2 = eventlet.patcher.original('threading').Thread(target=pthread2)
    real_thread2.start()
    real_thread1.join()
    real_thread2.join()
    self.assertEqual(thread_id, owner)
    self.assertIsNone(self.mutex.owner)