from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def normalize_subvolume_path(path):
    """
    Normalizes btrfs subvolume paths to ensure exactly one leading slash, no trailing slashes and no consecutive slashes.
    In addition, if the path is prefixed with a leading <FS_TREE>, this value is removed.
    """
    fstree_stripped = re.sub('^<FS_TREE>', '', path)
    result = re.sub('/+$', '', re.sub('/+', '/', '/' + fstree_stripped))
    return result if len(result) > 0 else '/'