import pytest
@pytest.mark.parametrize('punct', PUNCT_OPEN)
@pytest.mark.parametrize('text', ['Привет', 'Привіт', 'Ґелґотати', "З'єднання", 'Єдність', 'їхні'])
def test_uk_tokenizer_splits_same_open_punct(uk_tokenizer, punct, text):
    tokens = uk_tokenizer(punct + punct + punct + text)
    assert len(tokens) == 4
    assert tokens[0].text == punct
    assert tokens[3].text == text