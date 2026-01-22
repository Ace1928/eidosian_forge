import pytest
from spacy.tokens import Doc
def test_noun_chunks_is_parsed_pt(pt_tokenizer):
    """Test that noun_chunks raises Value Error for 'pt' language if Doc is not parsed."""
    doc = pt_tokenizer('en Oxford este verano')
    with pytest.raises(ValueError):
        list(doc.noun_chunks)