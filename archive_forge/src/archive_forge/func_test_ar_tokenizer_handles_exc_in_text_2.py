import pytest
def test_ar_tokenizer_handles_exc_in_text_2(ar_tokenizer):
    text = 'يبلغ طول مضيق طارق 14كم '
    tokens = ar_tokenizer(text)
    assert len(tokens) == 6