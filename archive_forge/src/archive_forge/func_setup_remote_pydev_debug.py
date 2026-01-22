import base64
import collections.abc
import contextlib
import grp
import hashlib
import itertools
import os
import pwd
import uuid
from cryptography import x509
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import urllib
from keystone.common import password_hashing
import keystone.conf
from keystone import exception
from keystone.i18n import _
def setup_remote_pydev_debug():
    if CONF.pydev_debug_host and CONF.pydev_debug_port:
        try:
            try:
                from pydev import pydevd
            except ImportError:
                import pydevd
            pydevd.settrace(CONF.pydev_debug_host, port=CONF.pydev_debug_port, stdoutToServer=True, stderrToServer=True)
            return True
        except Exception:
            LOG.exception('Error setting up the debug environment. Verify that the option --debug-url has the format <host>:<port> and that a debugger processes is listening on that port.')
            raise