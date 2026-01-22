import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('text', ['(', '((', '<'])
def test_tl_tokenizer_handles_only_punct(tl_tokenizer, text):
    tokens = tl_tokenizer(text)
    assert len(tokens) == len(text)