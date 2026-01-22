import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def import_refs(self, base: Ref, other: Dict[Ref, ObjectID], committer: Optional[bytes]=None, timestamp: Optional[bytes]=None, timezone: Optional[bytes]=None, message: Optional[bytes]=None, prune: bool=False):
    if prune:
        to_delete = set(self.subkeys(base))
    else:
        to_delete = set()
    for name, value in other.items():
        if value is None:
            to_delete.add(name)
        else:
            self.set_if_equals(b'/'.join((base, name)), None, value, message=message)
        if to_delete:
            try:
                to_delete.remove(name)
            except KeyError:
                pass
    for ref in to_delete:
        self.remove_if_equals(b'/'.join((base, ref)), None, message=message)