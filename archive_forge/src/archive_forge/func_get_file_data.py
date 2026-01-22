import hashlib
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Set, Tuple
from platformdirs import user_cache_dir
from _black_version import version as __version__
from black.mode import Mode
from black.output import err
@staticmethod
def get_file_data(path: Path) -> FileData:
    """Return file data for path."""
    stat = path.stat()
    hash = Cache.hash_digest(path)
    return FileData(stat.st_mtime, stat.st_size, hash)