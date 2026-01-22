from ...language import BaseDefaults, Language
from ...tokens import Doc
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
def thai_tokenizer_factory(nlp):
    return ThaiTokenizer(nlp.vocab)