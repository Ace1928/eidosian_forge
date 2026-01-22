import pytest
def test_ml_tokenizer_handles_long_text(ml_tokenizer):
    text = 'അനാവശ്യമായി കണ്ണിലും മൂക്കിലും വായിലും സ്പർശിക്കാതിരിക്കുക'
    tokens = ml_tokenizer(text)
    assert len(tokens) == 5