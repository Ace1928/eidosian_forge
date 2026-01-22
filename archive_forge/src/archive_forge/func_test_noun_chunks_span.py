import pytest
from spacy.tokens import Doc
def test_noun_chunks_span(doc, en_tokenizer):
    """Test that the span.noun_chunks property works correctly"""
    doc_chunks = list(doc.noun_chunks)
    span = doc[0:3]
    span_chunks = list(span.noun_chunks)
    assert 0 < len(span_chunks) < len(doc_chunks)
    for chunk in span_chunks:
        assert chunk in doc_chunks
        assert chunk.start >= 0
        assert chunk.end <= 3