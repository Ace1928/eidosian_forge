import cgi
import datetime
import inspect
import os
import re
import socket
import types
import unittest
import six
from six.moves import range  # pylint: disable=redefined-builtin
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
def get_module_name(module_attribute):
    """Get the module name.

    Args:
      module_attribute: An attribute of the module.

    Returns:
      The fully qualified module name or simple module name where
      'module_attribute' is defined if the module name is "__main__".
    """
    if module_attribute.__module__ == '__main__':
        module_file = inspect.getfile(module_attribute)
        default = os.path.basename(module_file).split('.')[0]
        return default
    return module_attribute.__module__