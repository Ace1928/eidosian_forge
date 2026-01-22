from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def cache_execution_id(execution_name):
    """Saves the execution resource to a named cache file.

  Args:
    execution_name: the execution resource name
  """
    try:
        files.WriteFileContents(_get_cache_path(), execution_name)
    except files.Error:
        pass