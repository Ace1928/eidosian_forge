from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def find_module_defining_flag(self, flagname, default=None):
    """Return the name of the module defining this flag, or default.

    Args:
      flagname: str, name of the flag to lookup.
      default: Value to return if flagname is not defined. Defaults to None.

    Returns:
      The name of the module which registered the flag with this name.
      If no such module exists (i.e. no flag with this name exists),
      we return default.
    """
    registered_flag = self._flags().get(flagname)
    if registered_flag is None:
        return default
    for module, flags in six.iteritems(self.flags_by_module_dict()):
        for flag in flags:
            if flag.name == registered_flag.name and flag.short_name == registered_flag.short_name:
                return module
    return default