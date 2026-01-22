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
def binary_encode(text, encoding='utf-8', errors='strict'):
    """Encodes a text string into a binary string using given encoding.

    Does nothing if data is already a binary string (raises on unknown types).
    """
    if isinstance(text, bytes):
        return text
    else:
        return encodeutils.safe_encode(text, encoding=encoding, errors=errors)