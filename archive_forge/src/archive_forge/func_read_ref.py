import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def read_ref(self, refname):
    """Read a reference without following any references.

        Args:
          refname: The name of the reference
        Returns: The contents of the ref file, or None if it does
            not exist.
        """
    contents = self.read_loose_ref(refname)
    if not contents:
        contents = self.get_packed_refs().get(refname, None)
    return contents