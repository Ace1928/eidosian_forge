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
def test_json_with_extra(self):
    test_msg = 'This is a %(test)s line'
    test_data = {'test': 'log'}
    extra_data = {'special_user': 'user1', 'special_tenant': 'unicorns'}
    self.log.debug(test_msg, test_data, key='value', extra=extra_data)
    data = jsonutils.loads(self.stream.getvalue())
    self.assertTrue(data)
    self.assertIn('extra', data)
    for k, v in extra_data.items():
        self.assertIn(k, data['extra'])
        self.assertEqual(v, data['extra'][k])