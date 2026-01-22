import contextlib
import os
import re
import textwrap
import time
from urllib import parse
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import prettytable
from novaclient import exceptions
from novaclient.i18n import _
def is_integer_like(val):
    """Returns validation of a value as an integer."""
    try:
        int(val)
        return True
    except (TypeError, ValueError, AttributeError):
        return False