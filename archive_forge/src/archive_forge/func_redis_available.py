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
def redis_available(min_version):
    client = redis.Redis()
    try:
        client.ping()
    except Exception:
        return False
    else:
        ok, redis_version = redis_utils.is_server_new_enough(client, min_version)
        return ok