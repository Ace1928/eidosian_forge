import abc
import errno
import functools
import os
import re
import signal
import struct
import subprocess
import sys
import time
from eventlet.green import socket
import eventlet.greenio
import eventlet.wsgi
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_concurrency import processutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from osprofiler import opts as profiler_opts
import routes.middleware
import webob.dec
import webob.exc
from webob import multidict
from glance.common import config
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
import glance.db
from glance import housekeeping
from glance import i18n
from glance.i18n import _, _LE, _LI, _LW
from glance import sqlite_migration
def translate_exception(req, e):
    """Translates all translatable elements of the given exception."""
    if not hasattr(req, 'best_match_language'):
        return e
    locale = req.best_match_language()
    if isinstance(e, webob.exc.HTTPError):
        e.explanation = i18n.translate(e.explanation, locale)
        e.detail = i18n.translate(e.detail, locale)
        if getattr(e, 'body_template', None):
            e.body_template = i18n.translate(e.body_template, locale)
    return e