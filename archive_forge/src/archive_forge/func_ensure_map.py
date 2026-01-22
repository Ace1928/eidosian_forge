import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
def ensure_map(m):
    if m is None:
        return {}
    elif isinstance(m, collections.abc.Mapping):
        return m
    else:
        msg = _('Incorrect arguments: to "%(fn_name)s", arguments must be a list of maps. Example:  %(example)s') % self.fmt_data
        raise TypeError(msg)