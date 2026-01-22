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
Test that serialization with custom tokenizer works without token_match.
    See: https://support.prodi.gy/t/how-to-save-a-custom-tokenizer/661/2
    