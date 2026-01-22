import time
import socket
import argparse
import sys
import itertools
import contextlib
import platform
from collections import abc
import urllib.parse
from tempora import timing
def occupied(host, port, timeout=float('Inf')):
    """
    Wait for the specified port to become occupied (accepting requests).
    Return when the port is occupied or raise a Timeout if timeout has
    elapsed.

    Timeout may be specified in seconds or as a timedelta.
    If timeout is None or ∞, the routine will run indefinitely.

    >>> occupied('localhost', find_available_local_port(), .1)
    Traceback (most recent call last):
    ...
    Timeout: Port ... not bound on localhost.

    >>> occupied(None, None)
    Traceback (most recent call last):
    ...
    ValueError: Host values of '' or None are not allowed.
    """
    if not host:
        raise ValueError("Host values of '' or None are not allowed.")
    timer = timing.Timer(timeout)
    while True:
        try:
            Checker(timeout=0.5).assert_free(host, port)
            if timer.expired():
                raise Timeout('Port {port} not bound on {host}.'.format(**locals()))
            time.sleep(0.1)
        except PortNotFree:
            return