import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_split_orths_mismatch(en_vocab):
    """Test that the regular retokenizer.split raises an error if the orths
    don't match the original token text. There might still be a method that
    allows this, but for the default use cases, merging and splitting should
    always conform with spaCy's non-destructive tokenization policy. Otherwise,
    it can lead to very confusing and unexpected results.
    """
    doc = Doc(en_vocab, words=['LosAngeles', 'start', '.'])
    with pytest.raises(ValueError):
        with doc.retokenize() as retokenizer:
            retokenizer.split(doc[0], ['L', 'A'], [(doc[0], 0), (doc[0], 0)])