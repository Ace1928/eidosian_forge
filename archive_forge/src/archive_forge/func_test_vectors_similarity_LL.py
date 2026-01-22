import numpy
import pytest
from spacy.tokens import Doc
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine
def test_vectors_similarity_LL(vocab, vectors):
    [(word1, vec1), (word2, vec2)] = vectors
    lex1 = vocab[word1]
    lex2 = vocab[word2]
    assert lex1.has_vector
    assert lex2.has_vector
    assert lex1.vector_norm != 0
    assert lex2.vector_norm != 0
    assert lex1.vector[0] != lex2.vector[0] and lex1.vector[1] != lex2.vector[1]
    assert isinstance(lex1.similarity(lex2), float)
    assert numpy.isclose(lex1.similarity(lex2), get_cosine(vec1, vec2))
    assert numpy.isclose(lex2.similarity(lex2), lex1.similarity(lex1))