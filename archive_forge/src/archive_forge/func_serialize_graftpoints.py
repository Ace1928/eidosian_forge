import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def serialize_graftpoints(graftpoints: Dict[bytes, List[bytes]]) -> bytes:
    """Convert a dictionary of grafts into string.

    The graft dictionary is:
        <commit sha1>: [<parent sha1>*]

    Each line is formatted as:
        <commit sha1> <parent sha1> [<parent sha1>]*

    https://git.wiki.kernel.org/index.php/GraftPoint

    """
    graft_lines = []
    for commit, parents in graftpoints.items():
        if parents:
            graft_lines.append(commit + b' ' + b' '.join(parents))
        else:
            graft_lines.append(commit)
    return b'\n'.join(graft_lines)