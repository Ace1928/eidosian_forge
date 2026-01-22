from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from calendar import timegm
import getpass
import logging
import os
import re
import time
import six
from gslib.exception import CommandException
from gslib.tz_utc import UTC
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import SECONDS_PER_DAY
Constructor for POSIXAttributes class which holds relevant data.

    Args:
      atime: The access time of the file/object.
      mtime: The modification time of the file/object.
      uid: The user ID that owns the file.
      gid: The group ID that the user is in.
      mode: An instance of POSIXMode.
    