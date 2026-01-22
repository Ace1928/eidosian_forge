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
def test_json_formatter(self):
    self.CONF(['--use-json'])
    log._setup_logging_from_conf(self.CONF, 'test', 'test')
    logger = log._loggers[None].logger
    for handler in logger.handlers:
        formatter = handler.formatter
        self.assertIsInstance(formatter, formatters.JSONFormatter)