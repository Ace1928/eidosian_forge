import collections
import logging
import threading
import time
import types
def result_iterator():
    try:
        fs.reverse()
        while fs:
            if timeout is None:
                yield _result_or_cancel(fs.pop())
            else:
                yield _result_or_cancel(fs.pop(), end_time - time.monotonic())
    finally:
        for future in fs:
            future.cancel()