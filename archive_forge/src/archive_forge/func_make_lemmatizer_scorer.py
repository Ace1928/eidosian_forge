import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from thinc.api import Model
from .. import util
from ..errors import Errors, Warnings
from ..language import Language
from ..lookups import Lookups, load_lookups
from ..scorer import Scorer
from ..tokens import Doc, Token
from ..training import Example
from ..util import SimpleFrozenList, logger, registry
from ..vocab import Vocab
from .pipe import Pipe
@registry.scorers('spacy.lemmatizer_scorer.v1')
def make_lemmatizer_scorer():
    return lemmatizer_score