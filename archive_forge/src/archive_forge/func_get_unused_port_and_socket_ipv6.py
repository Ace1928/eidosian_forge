import errno
import functools
import http.client
import http.server
import io
import os
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from unittest import mock
from alembic import command as alembic_command
import fixtures
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_config import fixture as cfg_fixture
from oslo_log.fixture import logging_error as log_fixture
from oslo_log import log
from oslo_utils import timeutils
from oslo_utils import units
import testtools
import webob
from glance.api.v2 import cached_images
from glance.common import config
from glance.common import exception
from glance.common import property_utils
from glance.common import utils
from glance.common import wsgi
from glance import context
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests.unit import fixtures as glance_fixtures
def get_unused_port_and_socket_ipv6():
    """
    Returns an unused port on localhost and the open socket
    from which it was created, but uses IPv6 (::1).
    """
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.bind(('::1', 0))
    addr, port, flowinfo, scopeid = s.getsockname()
    return (port, s)