import os
import time
import warnings
from typing import Iterator, cast
from .. import errors, pyutils, registry, trace
def create_archive(format, tree, name, root=None, subdir=None, force_mtime=None, recurse_nested=False) -> Iterator[bytes]:
    try:
        archive_fn = format_registry.get(format)
    except KeyError as exc:
        raise errors.NoSuchExportFormat(format) from exc
    return cast(Iterator[bytes], archive_fn(tree, name, root=root, subdir=subdir, force_mtime=force_mtime, recurse_nested=recurse_nested))