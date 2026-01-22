import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def available_backup_name(self, name):
    backup_name = osutils.available_backup_name(name, self.backup_exists)
    self.backups.append(backup_name)
    return backup_name