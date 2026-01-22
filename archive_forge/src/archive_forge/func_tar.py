import io
import os
import re
import tarfile
import tempfile
from .fnmatch import fnmatch
from ..constants import IS_WINDOWS_PLATFORM
def tar(path, exclude=None, dockerfile=None, fileobj=None, gzip=False):
    root = os.path.abspath(path)
    exclude = exclude or []
    dockerfile = dockerfile or (None, None)
    extra_files = []
    if dockerfile[1] is not None:
        dockerignore_contents = '\n'.join((exclude or ['.dockerignore']) + [dockerfile[0]])
        extra_files = [('.dockerignore', dockerignore_contents), dockerfile]
    return create_archive(files=sorted(exclude_paths(root, exclude, dockerfile=dockerfile[0])), root=root, fileobj=fileobj, gzip=gzip, extra_files=extra_files)