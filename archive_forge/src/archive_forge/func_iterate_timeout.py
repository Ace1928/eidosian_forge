from collections.abc import Mapping
import hashlib
import queue
import string
import threading
import time
import typing as ty
import keystoneauth1
from keystoneauth1 import adapter as ks_adapter
from keystoneauth1 import discover
from openstack import _log
from openstack import exceptions
def iterate_timeout(timeout, message, wait=2):
    """Iterate and raise an exception on timeout.

    This is a generator that will continually yield and sleep for
    wait seconds, and if the timeout is reached, will raise an exception
    with <message>.

    """
    log = _log.setup_logging('openstack.iterate_timeout')
    try:
        if wait is None:
            wait = 2
        elif wait == 0:
            wait = 0.1 if timeout is None else min(0.1, timeout)
        wait = float(wait)
    except ValueError:
        raise exceptions.SDKException('Wait value must be an int or float value. {wait} given instead'.format(wait=wait))
    start = time.time()
    count = 0
    while timeout is None or time.time() < start + timeout:
        count += 1
        yield count
        log.debug('Waiting %s seconds', wait)
        time.sleep(wait)
    raise exceptions.ResourceTimeout(message)