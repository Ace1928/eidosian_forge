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
def test_resource_key_in_log_msg(self):
    color = handlers.ColorHandler.LEVEL_COLORS[logging.INFO]
    ctxt = _fake_context()
    resource = 'resource-202260f9-1224-490d-afaf-6a744c13141f'
    fake_resource = {'name': resource}
    message = 'info'
    self.colorlog.info(message, context=ctxt, resource=fake_resource)
    expected = '%s [%s]: [%s] %s\x1b[00m\n' % (color, ctxt.request_id, resource, message)
    self.assertEqual(expected, self.stream.getvalue())