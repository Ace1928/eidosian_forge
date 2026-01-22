import pytest
def test_sa_tokenizer_handles_long_text(sa_tokenizer):
    text = 'नानाविधानि दिव्यानि नानावर्णाकृतीनि च।।'
    tokens = sa_tokenizer(text)
    assert len(tokens) == 6