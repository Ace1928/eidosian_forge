from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def get_cached_execution_id():
    """Gets the cached execution object.

  Returns:
    execution: the execution resource name
  """
    cache_path = _get_cache_path()
    if not os.path.isfile(cache_path):
        raise exceptions.Error(_NO_CACHE_MESSAGE)
    try:
        cached_execution = files.ReadFileContents(cache_path)
        execution_ref = resources.REGISTRY.Parse(cached_execution, collection=EXECUTION_COLLECTION)
        log.status.Print('Using cached execution name: {}'.format(execution_ref.RelativeName()))
        return execution_ref
    except Exception:
        raise exceptions.Error(_NO_CACHE_MESSAGE)