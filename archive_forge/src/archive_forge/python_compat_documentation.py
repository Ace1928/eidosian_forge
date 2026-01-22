from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
Generate all config files for the module.

    Returns:
      list(ext_runtime.GeneratedFile) A list of the config files
        that were generated
    