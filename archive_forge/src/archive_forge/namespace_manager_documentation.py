import os
import re
import warnings
from googlecloudsdk.third_party.appengine.api import lib_config
Raises an exception if value is not a valid Namespace string.

  A namespace must match the regular expression ([0-9A-Za-z._-]{0,100}).

  Args:
    value: string, the value to validate.
    exception: exception type to raise.
  