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
def main_module_help(self):
    """Describes the key flags of the main module.

    Returns:
      str, describing the key flags of the main module.
    """
    return self.module_help(sys.argv[0])