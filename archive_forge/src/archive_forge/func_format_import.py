import argparse
import collections
import importlib
import os
import sys
from tensorflow.python.tools.api.generator import doc_srcs
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
import sys as _sys
from tensorflow.python.util import module_wrapper as _module_wrapper
def format_import(self, source_module_name, source_name, dest_name):
    """Formats import statement.

    Args:
      source_module_name: (string) Source module to import from.
      source_name: (string) Source symbol name to import.
      dest_name: (string) Destination alias name.

    Returns:
      An import statement string.
    """
    if self._lazy_loading:
        return "  '%s': ('%s', '%s')," % (dest_name, source_module_name, source_name)
    elif source_module_name:
        if source_name == dest_name:
            return 'from %s import %s' % (source_module_name, source_name)
        else:
            return 'from %s import %s as %s' % (source_module_name, source_name, dest_name)
    elif source_name == dest_name:
        return 'import %s' % source_name
    else:
        return 'import %s as %s' % (source_name, dest_name)