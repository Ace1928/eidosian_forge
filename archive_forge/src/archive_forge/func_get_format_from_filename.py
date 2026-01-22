import os
import time
import warnings
from typing import Iterator, cast
from .. import errors, pyutils, registry, trace
def get_format_from_filename(self, filename):
    """Determine the archive format from an extension.

        :param filename: Filename to guess from
        :return: A format name, or None
        """
    for ext, format in self._extension_map.items():
        if filename.endswith(ext):
            return format
    else:
        return None