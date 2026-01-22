import os
import time
import yaml
from . import config, debug, errors, lock, osutils, ui, urlutils
from .decorators import only_raises
from .errors import (DirectoryNotEmpty, LockBreakMismatch, LockBroken,
from .i18n import gettext
from .osutils import format_delta, get_host_name, rand_chars
from .trace import mutter, note
from .transport import FileExists, NoSuchFile
def to_readable_dict(self):
    """Turn the holder info into a dict of human-readable attributes.

        For example, the start time is presented relative to the current time,
        rather than as seconds since the epoch.

        Returns a list of [user, hostname, pid, time_ago] all as readable
        strings.
        """
    start_time = self.info_dict.get('start_time')
    if start_time is None:
        time_ago = '(unknown)'
    else:
        time_ago = format_delta(time.time() - self.info_dict['start_time'])
    user = self.info_dict.get('user', '<unknown>')
    hostname = self.info_dict.get('hostname', '<unknown>')
    pid = self.info_dict.get('pid', '<unknown>')
    return dict(user=user, hostname=hostname, pid=pid, time_ago=time_ago)