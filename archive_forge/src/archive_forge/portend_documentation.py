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

        Assert that the given addr is free
        in that all attempts to connect fail within the timeout
        or raise a PortNotFree exception.

        >>> free_port = find_available_local_port()

        >>> Checker().assert_free('localhost', free_port)
        >>> Checker().assert_free('127.0.0.1', free_port)
        >>> Checker().assert_free('::1', free_port)

        Also accepts an addr tuple

        >>> addr = '::1', free_port, 0, 0
        >>> Checker().assert_free(addr)

        Host might refer to a server bind address like '::', which
        should use localhost to perform the check.

        >>> Checker().assert_free('::', free_port)
        