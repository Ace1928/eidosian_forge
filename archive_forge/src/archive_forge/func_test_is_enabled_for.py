from contextlib import contextmanager
import copy
import datetime
import io
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from unittest import mock
from dateutil import tz
from oslo_config import cfg
from oslo_config import fixture as fixture_config  # noqa
from oslo_context import context
from oslo_context import fixture as fixture_context
from oslo_i18n import fixture as fixture_trans
from oslo_serialization import jsonutils
from oslotest import base as test_base
import testtools
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
from oslo_log import log
from oslo_utils import units
def test_is_enabled_for(self):
    self.assertTrue(self.log.isEnabledFor(logging.INFO))
    self.assertFalse(self.log_no_debug.isEnabledFor(logging.DEBUG))
    self.assertTrue(self.log_below_debug.isEnabledFor(logging.DEBUG))
    self.assertTrue(self.log_below_debug.isEnabledFor(7))
    self.assertTrue(self.log_trace.isEnabledFor(log.TRACE))