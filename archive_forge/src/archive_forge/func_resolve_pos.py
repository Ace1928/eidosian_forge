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
def resolve_pos(orth, tag, next_tag):
    """If necessary, add a field to the POS tag for UD mapping.
    Under Universal Dependencies, sometimes the same Unidic POS tag can
    be mapped differently depending on the literal token or its context
    in the sentence. This function returns resolved POSs for both token
    and next_token by tuple.
    """
    if tag in TAG_ORTH_MAP:
        orth_map = TAG_ORTH_MAP[tag]
        if orth in orth_map:
            return (orth_map[orth], None)
    if next_tag:
        tag_bigram = (tag, next_tag)
        if tag_bigram in TAG_BIGRAM_MAP:
            current_pos, next_pos = TAG_BIGRAM_MAP[tag_bigram]
            if current_pos is None:
                return (TAG_MAP[tag][POS], next_pos)
            else:
                return (current_pos, next_pos)
    return (TAG_MAP[tag][POS], None)