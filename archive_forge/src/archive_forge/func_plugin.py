import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
@property
def plugin(self):
    from breezy.plugin import get_loaded_plugin
    return get_loaded_plugin(self.plugin_name)