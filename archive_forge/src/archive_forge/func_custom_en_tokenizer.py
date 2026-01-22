import re
import pytest
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
@pytest.fixture
def custom_en_tokenizer(en_vocab):
    prefix_re = compile_prefix_regex(English.Defaults.prefixes)
    suffix_re = compile_suffix_regex(English.Defaults.suffixes)
    custom_infixes = ['\\.\\.\\.+', '(?<=[0-9])-(?=[0-9])', '[0-9]+(,[0-9]+)+', '[\\[\\]!&:,()\\*—–\\/-]']
    infix_re = compile_infix_regex(custom_infixes)
    token_match_re = re.compile('a-b')
    return Tokenizer(en_vocab, English.Defaults.tokenizer_exceptions, prefix_re.search, suffix_re.search, infix_re.finditer, token_match=token_match_re.match)