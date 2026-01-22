import pickle
import re
import pytest
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.training import Example
from spacy.util import load_config_from_str
from ..util import make_tempdir
def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, {}, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer)