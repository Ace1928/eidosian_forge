import pytest
@pytest.mark.parametrize('text', ['test@example.com', 'john.doe@example.co.uk'])
@pytest.mark.issue(1698)
def test_issue1698(en_tokenizer, text):
    """Test that doc doesn't identify email-addresses as URLs"""
    doc = en_tokenizer(text)
    assert len(doc) == 1
    assert not doc[0].like_url