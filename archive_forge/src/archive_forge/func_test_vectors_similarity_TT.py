import numpy
import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine
def test_vectors_similarity_TT(vocab, vectors):
    [(word1, vec1), (word2, vec2)] = vectors
    doc = Doc(vocab, words=[word1, word2])
    assert doc[0].has_vector
    assert doc[1].has_vector
    assert doc[0].vector_norm != 0
    assert doc[1].vector_norm != 0
    assert doc[0].vector[0] != doc[1].vector[0] and doc[0].vector[1] != doc[1].vector[1]
    assert isinstance(doc[0].similarity(doc[1]), float)
    assert numpy.isclose(doc[0].similarity(doc[1]), get_cosine(vec1, vec2))
    assert numpy.isclose(doc[1].similarity(doc[0]), doc[0].similarity(doc[1]))