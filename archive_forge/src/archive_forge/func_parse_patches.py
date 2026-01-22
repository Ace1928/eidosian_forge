import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def parse_patches(iter_lines, allow_dirty=False, keep_dirty=False):
    """
    :arg iter_lines: iterable of lines to parse for patches
    :kwarg allow_dirty: If True, allow text that's not part of the patch at
        selected places.  This includes comments before and after a patch
        for instance.  Default False.
    :kwarg keep_dirty: If True, returns a dict of patches with dirty headers.
        Default False.
    """
    for patch_lines in iter_file_patch(iter_lines, allow_dirty, keep_dirty):
        if 'dirty_head' in patch_lines:
            yield {'patch': parse_patch(patch_lines['saved_lines'], allow_dirty), 'dirty_head': patch_lines['dirty_head']}
        else:
            yield parse_patch(patch_lines, allow_dirty)