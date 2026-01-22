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
@contextmanager
def mutate_conf(self, conf1, conf2):
    loginis = self.create_tempfiles([('log1.ini', self.mk_log_config(conf1)), ('log2.ini', self.mk_log_config(conf2))])
    confs = self.setup_confs('[DEFAULT]\nlog_config_append = %s\n' % loginis[0], '[DEFAULT]\nlog_config_append = %s\n' % loginis[1])
    log.setup(self.CONF, '')
    yield (loginis, confs)
    shutil.copy(confs[1], confs[0])
    os.utime(self.CONF.log_config_append, (0, 0))
    self.CONF.mutate_config_files()