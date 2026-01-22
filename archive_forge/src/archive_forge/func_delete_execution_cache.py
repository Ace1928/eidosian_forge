from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def delete_execution_cache():
    """Clears the execution cache.

  Returns:
    bool: True if the file was found and deleted, false otherwise.
  """
    try:
        os.remove(_get_cache_path())
    except OSError:
        return False
    return True