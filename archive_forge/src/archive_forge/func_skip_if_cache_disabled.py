import atexit
import base64
import contextlib
import datetime
import functools
import hashlib
import json
import secrets
import ldap
import os
import shutil
import socket
import sys
import uuid
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography import x509
import fixtures
import flask
from flask import testing as flask_testing
import http.client
from oslo_config import fixture as config_fixture
from oslo_context import context as oslo_context
from oslo_context import fixture as oslo_ctx_fixture
from oslo_log import fixture as log_fixture
from oslo_log import log
from oslo_utils import timeutils
import testtools
from testtools import testcase
import keystone.api
from keystone.common import context
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common as ks_ldap
from keystone import notifications
from keystone.resource.backends import base as resource_base
from keystone.server.flask import application as flask_app
from keystone.server.flask import core as keystone_flask
from keystone.tests.unit import ksfixtures
def skip_if_cache_disabled(*sections):
    """Skip a test if caching is disabled, this is a decorator.

    Caching can be disabled either globally or for a specific section.

    In the code fragment::

        @skip_if_cache_is_disabled('assignment', 'token')
        def test_method(*args):
            ...

    The method test_method would be skipped if caching is disabled globally via
    the `enabled` option in the `cache` section of the configuration or if
    the `caching` option is set to false in either `assignment` or `token`
    sections of the configuration.  This decorator can be used with no
    arguments to only check global caching.

    If a specified configuration section does not define the `caching` option,
    this decorator makes the caching enabled if `enabled` option in the `cache`
    section of the configuration is true.

    """

    def wrapper(f):

        @functools.wraps(f)
        def inner(*args, **kwargs):
            if not CONF.cache.enabled:
                raise testcase.TestSkipped('Cache globally disabled.')
            for s in sections:
                conf_sec = getattr(CONF, s, None)
                if conf_sec is not None:
                    if not getattr(conf_sec, 'caching', True):
                        raise testcase.TestSkipped('%s caching disabled.' % s)
            return f(*args, **kwargs)
        return inner
    return wrapper