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
@mock.patch('oslo_log.watchers.FastWatchedFileHandler')
@mock.patch('oslo_log.log._get_log_file_path', return_value='test.conf')
@mock.patch('platform.system', return_value='Linux')
def test_watchlog_on_linux(self, platfotm_mock, path_mock, handler_mock):
    self.config(watch_log_file=True)
    log._setup_logging_from_conf(self.CONF, 'test', 'test')
    handler_mock.assert_called_once_with(path_mock.return_value)
    self.assertEqual(self.log_handlers[0], handler_mock.return_value)