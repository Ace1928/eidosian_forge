import warnings
from functools import partial
from pathlib import Path
from typing import (
import srsly
from .. import util
from ..errors import Errors, Warnings
from ..language import Language
from ..matcher import Matcher, PhraseMatcher
from ..matcher.levenshtein import levenshtein_compare
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example
from ..util import SimpleFrozenList, ensure_path, registry
from .pipe import Pipe
@property
def ids(self) -> Tuple[str, ...]:
    """All IDs present in the match patterns.

        RETURNS (set): The string IDs.

        DOCS: https://spacy.io/api/spanruler#ids
        """
    return tuple(sorted(set([cast(str, p.get('id')) for p in self._patterns]) - set([None])))