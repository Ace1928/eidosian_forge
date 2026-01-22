import pytest
@pytest.mark.parametrize('wo_punct,w_punct', [("We've", "`We've"), ("couldn't", "couldn't)")])
def test_en_tokenizer_splits_defined_punct(en_tokenizer, wo_punct, w_punct):
    tokens = en_tokenizer(wo_punct)
    assert len(tokens) == 2
    tokens = en_tokenizer(w_punct)
    assert len(tokens) == 3