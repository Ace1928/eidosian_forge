import sys
import pytest
@pytest.mark.parametrize('text,length', [('108)', 2), ('XDN', 1)])
def test_tokenizer_excludes_false_pos_emoticons(tokenizer, text, length):
    tokens = tokenizer(text)
    assert len(tokens) == length