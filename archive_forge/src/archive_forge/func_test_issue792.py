import pytest
@pytest.mark.issue(792)
@pytest.mark.parametrize('text', ['This is a string ', 'This is a string '])
def test_issue792(en_tokenizer, text):
    """Test for Issue #792: Trailing whitespace is removed after tokenization."""
    doc = en_tokenizer(text)
    assert ''.join([token.text_with_ws for token in doc]) == text