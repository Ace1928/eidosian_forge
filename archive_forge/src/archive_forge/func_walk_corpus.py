import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Union
import srsly
from .. import util
from ..errors import Errors, Warnings
from ..tokens import Doc, DocBin
from ..vocab import Vocab
from .augment import dont_augment
from .example import Example
def walk_corpus(path: Union[str, Path], file_type) -> List[Path]:
    path = util.ensure_path(path)
    if not path.is_dir() and path.parts[-1].endswith(file_type):
        return [path]
    orig_path = path
    paths = [path]
    locs = []
    seen = set()
    for path in paths:
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.parts and path.parts[-1].startswith('.'):
            continue
        elif path.is_dir():
            paths.extend(path.iterdir())
        elif path.parts[-1].endswith(file_type):
            locs.append(path)
    if len(locs) == 0:
        warnings.warn(Warnings.W090.format(path=orig_path, format=file_type))
    locs.sort()
    return locs