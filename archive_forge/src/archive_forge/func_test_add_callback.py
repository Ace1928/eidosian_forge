import asyncio
import functools
import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_add_callback(self):

    def callback(stream, status, event):
        event.set()
    stream = cuda.stream()
    callback_event = threading.Event()
    stream.add_callback(callback, callback_event)
    self.assertTrue(callback_event.wait(1.0))