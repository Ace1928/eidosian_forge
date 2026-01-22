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
def pretty_choice_dict(values):
    """Returns a formatted dict as 'key=value'."""
    return pretty_choice_list(['%s=%s' % (k, values[k]) for k in sorted(values)])