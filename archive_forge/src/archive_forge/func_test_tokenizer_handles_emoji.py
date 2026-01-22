import sys
import pytest
@pytest.mark.parametrize('text,length', [('can you still dunk?🍕🍔😵LOL', 8), ('i💙you', 3), ('🤘🤘yay!', 4)])
def test_tokenizer_handles_emoji(tokenizer, text, length):
    if sys.maxunicode >= 1114111:
        tokens = tokenizer(text)
        assert len(tokens) == length