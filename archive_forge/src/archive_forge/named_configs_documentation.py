from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import threading
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import properties_file
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
Invalidate the cached property values.

    Args:
      mark_changed: bool, True if we are invalidating because we persisted
        a change to the installation config, the active configuration, or
        changed the active configuration. If so, the config sentinel is touched.
    