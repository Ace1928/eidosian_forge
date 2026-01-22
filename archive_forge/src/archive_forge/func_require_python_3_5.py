from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import optimize_parameters_util
from googlecloudsdk.core import properties
def require_python_3_5():
    """Task execution assumes Python versions >=3.5.

  Raises:
    InvalidPythonVersionError: if the Python version is not 3.5+.
  """
    if sys.version_info.major < 3 or (sys.version_info.major == 3 and sys.version_info.minor < 5):
        raise errors.InvalidPythonVersionError('This functionality does not support Python {}.{}.{}. Please upgrade to Python 3.5 or greater.'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro))