from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
from boto import config
import gslib
from gslib import exception
from gslib.utils import boto_util
from gslib.utils import execution_util
def get_context_config():
    """Retrieves ContextConfig global singleton.

  Returns:
    ContextConfig or None if global singleton doesn't exist.
  """
    return _singleton_config