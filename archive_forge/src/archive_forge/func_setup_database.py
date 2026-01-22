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
@mock.patch('oslo_db.sqlalchemy.enginefacade.writer.get_engine')
def setup_database(self, mock_get_engine):
    """Configure and prepare a fresh sqlite database."""
    db_file = 'sqlite:///%s/test.db' % self.test_dir
    self.config(connection=db_file, group='database')
    db_api.clear_db_env()
    engine = db_api.get_engine()
    mock_get_engine.return_value = engine
    with mock.patch('logging.config'):
        test_utils.db_sync(engine=engine)