from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import optimize_parameters_util
from googlecloudsdk.core import properties
Task execution assumes Python versions >=3.5.

  Raises:
    InvalidPythonVersionError: if the Python version is not 3.5+.
  