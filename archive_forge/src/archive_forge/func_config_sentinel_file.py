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
def config_sentinel_file(self):
    """Gets the path to the config sentinel.

    The sentinel is a file that we touch any time there is a change to config.
    External tools can check this file to see if they need to re-query gcloud's
    credential/config helper to get updated configuration information. Nothing
    is ever written to this file, it's timestamp indicates the last time config
    was changed.

    This does not take into account config changes made through environment
    variables as they are transient by nature. There is also the edge case of
    when a user updated installation config. That user's sentinel will be
    updated but other will not be.

    Returns:
      str, The path to the sentinel file.
    """
    return os.path.join(self.global_config_dir, 'config_sentinel')