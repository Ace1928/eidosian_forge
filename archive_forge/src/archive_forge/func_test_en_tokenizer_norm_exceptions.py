import pytest
@pytest.mark.parametrize('text,norms', [("I'm", ['i', 'am']), ("shan't", ['shall', 'not']), ("Many factors cause cancer 'cause it is complex", ['many', 'factors', 'cause', 'cancer', 'because', 'it', 'is', 'complex'])])
def test_en_tokenizer_norm_exceptions(en_tokenizer, text, norms):
    tokens = en_tokenizer(text)
    assert [token.norm_ for token in tokens] == norms