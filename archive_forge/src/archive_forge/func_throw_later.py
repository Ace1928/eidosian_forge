from collections import deque
from threading import local
def throw_later(self, reason, scheduler):

    def fn():
        raise reason
    scheduler.call(fn)