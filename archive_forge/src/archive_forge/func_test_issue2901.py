import pytest
from spacy.lang.ja import DetailedToken, Japanese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.issue(2901)
def test_issue2901():
    """Test that `nlp` doesn't fail."""
    try:
        nlp = Japanese()
    except ImportError:
        pytest.skip()
    doc = nlp('pythonが大好きです')
    assert doc