import pytest
@pytest.mark.parametrize('punct', PUNCT_CLOSE)
@pytest.mark.parametrize('text', ['Привет', 'Привіт', 'Ґелґотати', "З'єднання", 'Єдність', 'їхні'])
def test_uk_tokenizer_splits_same_close_punct(uk_tokenizer, punct, text):
    tokens = uk_tokenizer(text + punct + punct + punct)
    assert len(tokens) == 4
    assert tokens[0].text == text
    assert tokens[1].text == punct