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
def start_servers(self, **kwargs):
    """
        Starts the API and Registry servers (glance-control api start
        ) on unused ports.  glance-control
        should be installed into the python path

        Any kwargs passed to this method will override the configuration
        value in the conf file used in starting the servers.
        """
    self.cleanup()
    self.start_with_retry(self.api_server_multiple_backend, 'api_port', 3, **kwargs)
    if self.include_scrubber:
        exitcode, out, err = self.scrubber_daemon.start(**kwargs)
        self.assertEqual(0, exitcode, 'Failed to spin up the Scrubber daemon. Got: %s' % err)