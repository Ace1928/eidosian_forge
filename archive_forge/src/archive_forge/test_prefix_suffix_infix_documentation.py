import pytest
from spacy.lang.char_classes import ALPHA
from spacy.lang.punctuation import TOKENIZER_INFIXES
from spacy.language import BaseDefaults, Language
Allow zero-width 'infix' token during the tokenization process.