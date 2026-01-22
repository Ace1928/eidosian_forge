import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
import srsly
from ... import util
from ...errors import Errors, Warnings
from ...language import BaseDefaults, Language
from ...scorer import Scorer
from ...tokens import Doc
from ...training import Example, validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
def save_pkuseg_processors(path):
    if self.pkuseg_seg:
        data = (_get_pkuseg_trie_data(self.pkuseg_seg.preprocesser.trie), self.pkuseg_seg.postprocesser.do_process, sorted(list(self.pkuseg_seg.postprocesser.common_words)), sorted(list(self.pkuseg_seg.postprocesser.other_words)))
        srsly.write_msgpack(path, data)