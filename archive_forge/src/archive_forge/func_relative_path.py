from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import getpass
import io
import locale
import os
import platform as system_platform
import re
import ssl
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import http_proxy_setup
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import requests
import six
import urllib3
@property
def relative_path(self):
    """Returns path of log file relative to log directory, or the full path.

    Returns the full path when the log file is not *in* the log directory.

    Returns:
      str, the relative or full path of log file.
    """
    logs_dir = config.Paths().logs_dir
    if logs_dir is None:
        return self.filename
    rel_path = os.path.relpath(self.filename, config.Paths().logs_dir)
    if rel_path.startswith(os.path.pardir + os.path.sep):
        return self.filename
    return rel_path