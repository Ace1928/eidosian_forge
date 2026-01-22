import collections.abc
import contextlib
import datetime
import functools
import inspect
import io
import os
import re
import socket
import sys
import threading
import types
import enum
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import netutils
from oslo_utils import reflection
from taskflow.types import failure
def countdown_iter(start_at, decr=1):
    """Generator that decrements after each generation until <= zero.

    NOTE(harlowja): we can likely remove this when we can use an
    ``itertools.count`` that takes a step (on py2.6 which we still support
    that step parameter does **not** exist and therefore can't be used).
    """
    if decr <= 0:
        raise ValueError('Decrement value must be greater than zero and not %s' % decr)
    while start_at > 0:
        yield start_at
        start_at -= decr