import re
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import srsly
from thinc.api import Model
from ... import util
from ...errors import Errors
from ...language import BaseDefaults, Language
from ...pipeline import Morphologizer
from ...pipeline.morphologizer import DEFAULT_MORPH_MODEL
from ...scorer import Scorer
from ...symbols import POS
from ...tokens import Doc, MorphAnalysis
from ...training import validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .stop_words import STOP_WORDS
from .syntax_iterators import SYNTAX_ITERATORS
from .tag_bigram_map import TAG_BIGRAM_MAP
from .tag_map import TAG_MAP
from .tag_orth_map import TAG_ORTH_MAP
def from_disk(self, path: Union[str, Path], **kwargs) -> 'JapaneseTokenizer':
    path = util.ensure_path(path)
    serializers = {'cfg': lambda p: self._set_config(srsly.read_json(p))}
    util.from_disk(path, serializers, [])
    self.tokenizer = try_sudachi_import(self.split_mode)
    return self