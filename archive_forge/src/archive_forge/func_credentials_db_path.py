from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
@property
def credentials_db_path(self):
    """Gets the path to the file to store credentials in.

    This is generic key/value store format using sqlite.

    Returns:
      str, The path to the credential db file.
    """
    return os.path.join(self.global_config_dir, 'credentials.db')