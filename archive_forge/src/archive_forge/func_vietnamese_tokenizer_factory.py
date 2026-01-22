import re
import string
from pathlib import Path
from typing import Any, Dict, Union
import srsly
from ... import util
from ...language import BaseDefaults, Language
from ...tokens import Doc
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
def vietnamese_tokenizer_factory(nlp):
    return VietnameseTokenizer(nlp.vocab, use_pyvi=use_pyvi)