import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
def test_tl_tokenizer_splits_bracket_period(tl_tokenizer):
    text = '(Dumating siya kahapon).'
    tokens = tl_tokenizer(text)
    assert tokens[len(tokens) - 1].text == '.'