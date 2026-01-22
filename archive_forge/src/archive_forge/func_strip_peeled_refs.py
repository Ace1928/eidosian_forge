import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def strip_peeled_refs(refs):
    """Remove all peeled refs."""
    return {ref: sha for ref, sha in refs.items() if not ref.endswith(PEELED_TAG_SUFFIX)}