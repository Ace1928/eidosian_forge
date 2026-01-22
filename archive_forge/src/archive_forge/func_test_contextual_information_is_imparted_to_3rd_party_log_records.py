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
def test_contextual_information_is_imparted_to_3rd_party_log_records(self):
    ctxt = _fake_context()
    sa_log = logging.getLogger('sqlalchemy.engine')
    sa_log.setLevel(logging.INFO)
    message = 'emulate logging within sqlalchemy'
    sa_log.info(message)
    expected = 'HAS CONTEXT [%s]: %s\n' % (ctxt.request_id, message)
    self.assertEqual(expected, self.stream.getvalue())