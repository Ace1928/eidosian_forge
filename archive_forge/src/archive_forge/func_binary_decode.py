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
def binary_decode(data, encoding='utf-8', errors='strict'):
    """Decodes a binary string into a text string using given encoding.

    Does nothing if data is already a text string (raises on unknown types).
    """
    if isinstance(data, str):
        return data
    else:
        return encodeutils.safe_decode(data, incoming=encoding, errors=errors)