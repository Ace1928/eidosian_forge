import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
def pack_loose_objects(self):
    """Pack loose objects.

        Returns: Number of objects packed
        """
    objects = set()
    for sha in self._iter_loose_objects():
        objects.add((self._get_loose_object(sha), None))
    self.add_objects(list(objects))
    for obj, path in objects:
        self._remove_loose_object(obj.id)
    return len(objects)