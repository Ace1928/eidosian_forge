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
def prepare_query_string(params):
    """Convert dict params to query string"""
    if not params:
        return ''
    params = sorted(params.items(), key=lambda x: x[0])
    return '?%s' % parse.urlencode(params) if params else ''