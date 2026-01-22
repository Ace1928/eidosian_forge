import os.path
import re
from contextlib import suppress
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Union
from .config import Config, get_xdg_config_home_path
def find_matching(self, path: str) -> Iterable[Pattern]:
    """Find matching patterns for path.

        Args:
          path: Path to check
        Returns:
          Iterator over Pattern instances
        """
    if os.path.isabs(path):
        raise ValueError('%s is an absolute path' % path)
    filters = [(0, f) for f in self._global_filters]
    if os.path.sep != '/':
        path = path.replace(os.path.sep, '/')
    parts = path.split('/')
    matches = []
    for i in range(len(parts) + 1):
        dirname = '/'.join(parts[:i])
        for s, f in filters:
            relpath = '/'.join(parts[s:i])
            if i < len(parts):
                relpath += '/'
            matches += list(f.find_matching(relpath))
        ignore_filter = self._load_path(dirname)
        if ignore_filter is not None:
            filters.insert(0, (i, ignore_filter))
    return iter(matches)