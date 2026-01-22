from __future__ import annotations
import itertools
import fnmatch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ..compilers import lang_suffixes
from ..mesonlib import quiet_git
import typing as T
def parse_pattern_file(fname: Path) -> T.List[str]:
    patterns = []
    try:
        with fname.open(encoding='utf-8') as f:
            for line in f:
                pattern = line.strip()
                if pattern and (not pattern.startswith('#')):
                    patterns.append(pattern)
    except FileNotFoundError:
        pass
    return patterns