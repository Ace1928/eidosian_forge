import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def parse_patch(iter_lines, allow_dirty=False):
    """
    :arg iter_lines: iterable of lines to parse
    :kwarg allow_dirty: If True, allow the patch to have trailing junk.
        Default False
    """
    iter_lines = iter_lines_handle_nl(iter_lines)
    try:
        (orig_name, orig_ts), (mod_name, mod_ts) = get_patch_names(iter_lines)
    except BinaryFiles as e:
        return BinaryPatch(e.orig_name, e.mod_name)
    else:
        patch = Patch(orig_name, mod_name, orig_ts, mod_ts)
        for hunk in iter_hunks(iter_lines, allow_dirty):
            patch.hunks.append(hunk)
        return patch