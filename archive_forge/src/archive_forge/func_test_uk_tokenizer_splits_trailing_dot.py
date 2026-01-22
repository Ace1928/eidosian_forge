import pytest
@pytest.mark.parametrize('text', ['Привет.', 'Привіт.', 'Ґелґотати.', "З'єднання.", 'Єдність.', 'їхні.'])
def test_uk_tokenizer_splits_trailing_dot(uk_tokenizer, text):
    tokens = uk_tokenizer(text)
    assert tokens[1].text == '.'