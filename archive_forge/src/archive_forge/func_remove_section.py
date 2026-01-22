from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def remove_section(self, section):
    """Remove a file section."""
    existed = section in self._sections
    if existed:
        del self._sections[section]
        del self._proxies[section]
    return existed