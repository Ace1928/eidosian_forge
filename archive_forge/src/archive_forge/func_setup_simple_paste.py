import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
def setup_simple_paste(self):
    """Setup a very simple no-auth paste pipeline.

        This configures the API to be very direct, including only the
        middleware absolutely required for consistent API calls.
        """
    self.paste_config = os.path.join(self.test_dir, 'glance-api-paste.ini')
    with open(self.paste_config, 'w') as f:
        f.write(textwrap.dedent('\n            [filter:context]\n            paste.filter_factory = glance.api.middleware.context:                ContextMiddleware.factory\n            [filter:fakeauth]\n            paste.filter_factory = glance.tests.utils:                FakeAuthMiddleware.factory\n            [filter:cache]\n            paste.filter_factory = glance.api.middleware.cache:            CacheFilter.factory\n            [filter:cachemanage]\n            paste.filter_factory = glance.api.middleware.cache_manage:            CacheManageFilter.factory\n            [pipeline:glance-api-cachemanagement]\n            pipeline = context cache cachemanage rootapp\n            [pipeline:glance-api-caching]\n            pipeline = context cache rootapp\n            [pipeline:glance-api]\n            pipeline = context rootapp\n            [composite:rootapp]\n            paste.composite_factory = glance.api:root_app_factory\n            /v2: apiv2app\n            [app:apiv2app]\n            paste.app_factory = glance.api.v2.router:API.factory\n            '))