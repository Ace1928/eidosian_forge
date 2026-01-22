import contextlib
import string
import threading
import time
from oslo_utils import timeutils
import redis
from taskflow import exceptions
from taskflow.listeners import capturing
from taskflow.persistence.backends import impl_memory
from taskflow import retry
from taskflow import task
from taskflow.types import failure
from taskflow.utils import kazoo_utils
from taskflow.utils import redis_utils
def make_many(amount, task_cls=DummyTask, offset=0):
    name_pool = string.ascii_lowercase + string.ascii_uppercase
    tasks = []
    while amount > 0:
        if offset >= len(name_pool):
            raise AssertionError('Name pool size to small (%s < %s)' % (len(name_pool), offset + 1))
        tasks.append(task_cls(name=name_pool[offset]))
        offset += 1
        amount -= 1
    return tasks