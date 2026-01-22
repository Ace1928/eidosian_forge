import pytest
@pytest.mark.issue(1758)
def test_issue1758(en_tokenizer):
    """Test that "would've" is handled by the English tokenizer exceptions."""
    tokens = en_tokenizer("would've")
    assert len(tokens) == 2