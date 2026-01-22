import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def disable_plugins(state=None):
    """Disable loading plugins.

    Future calls to load_plugins() will be ignored.

    Args:
      state: The library state object that records loaded plugins.
    """
    if state is None:
        state = breezy.get_global_state()
    state.plugins = {}