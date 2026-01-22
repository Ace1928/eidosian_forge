import pytest
from spacy.tokens import Doc
def test_noun_chunks_is_parsed(la_tokenizer):
    """Test that noun_chunks raises Value Error for 'la' language if Doc is not parsed.
    To check this test, we're constructing a Doc
    with a new Vocab here and forcing is_parsed to 'False'
    to make sure the noun chunks don't run.
    """
    doc = la_tokenizer('Haec est sententia.')
    with pytest.raises(ValueError):
        list(doc.noun_chunks)