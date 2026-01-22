import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
def split_filter_op(expression):
    """Split operator from threshold in an expression.
    Designed for use on a comparative-filtering query field.
    When no operator is found, default to an equality comparison.

    :param expression: the expression to parse

    :returns: a tuple (operator, threshold) parsed from expression
    """
    left, sep, right = expression.partition(':')
    if sep:
        try:
            timeutils.parse_isotime(expression)
            op = 'eq'
            threshold = expression
        except ValueError:
            op = left
            threshold = right
    else:
        op = 'eq'
        threshold = left
    return (op, threshold)