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
def filtered_cached(self, sources: Iterable[Path]) -> Tuple[Set[Path], Set[Path]]:
    """Split an iterable of paths in `sources` into two sets.

        The first contains paths of files that modified on disk or are not in the
        cache. The other contains paths to non-modified files.
        """
    changed: Set[Path] = set()
    done: Set[Path] = set()
    for src in sources:
        if self.is_changed(src):
            changed.add(src)
        else:
            done.add(src)
    return (changed, done)