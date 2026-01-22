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
def skip_body_lines(start_line, report_lines):
    curr_line = start_line
    while len(report_lines[curr_line]) == 0 or report_lines[curr_line][0] != '=':
        curr_line += 1
    return curr_line