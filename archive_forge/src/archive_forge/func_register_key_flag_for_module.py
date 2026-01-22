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
def register_key_flag_for_module(self, module_name, flag):
    """Specifies that a flag is a key flag for a module.

    Args:
      module_name: str, the name of a Python module.
      flag: Flag, the Flag instance that is key to the module.
    """
    key_flags_by_module = self.key_flags_by_module_dict()
    key_flags = key_flags_by_module.setdefault(module_name, [])
    if flag not in key_flags:
        key_flags.append(flag)