from __future__ import annotations
import copy
import json
from .nbbase import from_dict
from .rwbase import NotebookReader, NotebookWriter, rejoin_lines, restore_bytes, split_lines
def writes(self, nb, **kwargs):
    """Convert a notebook object to a string."""
    kwargs['cls'] = BytesEncoder
    kwargs['indent'] = 1
    kwargs['sort_keys'] = True
    if kwargs.pop('split_lines', True):
        nb = split_lines(copy.deepcopy(nb))
    return json.dumps(nb, **kwargs)