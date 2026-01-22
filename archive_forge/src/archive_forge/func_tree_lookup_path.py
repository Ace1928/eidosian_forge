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
def tree_lookup_path(lookup_obj, root_sha, path):
    """Look up an object in a Git tree.

    Args:
      lookup_obj: Callback for retrieving object by SHA1
      root_sha: SHA1 of the root tree
      path: Path to lookup
    Returns: A tuple of (mode, SHA) of the resulting path.
    """
    tree = lookup_obj(root_sha)
    if not isinstance(tree, Tree):
        raise NotTreeError(root_sha)
    return tree.lookup_path(lookup_obj, path)