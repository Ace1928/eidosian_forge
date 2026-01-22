import datetime
import io
import os
import re
import signal
import sys
import threading
from unittest import mock
import fixtures
import greenlet
from oslotest import base
import oslo_config
from oslo_config import fixture
from oslo_reports import guru_meditation_report as gmr
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import opts
@mock.patch('oslo_utils.timeutils.utcnow', return_value=datetime.datetime(2014, 1, 1, 12, 0, 0))
def test_register_autorun_log_dir(self, mock_strtime):
    log_dir = self.useFixture(fixtures.TempDir()).path
    gmr.TextGuruMeditation.setup_autorun(FakeVersionObj(), 'fake-service', log_dir)
    os.kill(os.getpid(), signal.SIGUSR2)
    with open(os.path.join(log_dir, 'fake-service_gurumeditation_20140101120000')) as df:
        self.assertIn('Guru Meditation', df.read())