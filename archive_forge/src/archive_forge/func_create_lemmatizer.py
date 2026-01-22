from typing import Callable, Optional
from thinc.api import Model
from ...attrs import LANG
from ...language import BaseDefaults, Language
from ...lookups import Lookups
from ...util import update_exc
from ..tokenizer_exceptions import BASE_EXCEPTIONS
from .lemmatizer import MacedonianLemmatizer
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
@classmethod
def create_lemmatizer(cls, nlp=None, lookups=None):
    if lookups is None:
        lookups = Lookups()
    return MacedonianLemmatizer(lookups)