import numpy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.lang.en import English
from spacy.strings import hash_string  # type: ignore
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training.initialize import convert_vectors
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_cosine, make_tempdir
def test_vectors_attr():
    data = numpy.asarray([[0, 0, 0], [1, 2, 3], [9, 8, 7]], dtype='f')
    nlp = English()
    nlp.vocab.vectors = Vectors(data=data, keys=['A', 'B', 'C'])
    assert nlp.vocab.strings['A'] in nlp.vocab.vectors.key2row
    assert nlp.vocab.strings['a'] not in nlp.vocab.vectors.key2row
    assert nlp.vocab['A'].has_vector is True
    assert nlp.vocab['a'].has_vector is False
    assert nlp('A')[0].has_vector is True
    assert nlp('a')[0].has_vector is False
    nlp = English()
    nlp.vocab.vectors = Vectors(data=data, keys=['a', 'b', 'c'], attr='LOWER')
    assert nlp.vocab.strings['A'] not in nlp.vocab.vectors.key2row
    assert nlp.vocab.strings['a'] in nlp.vocab.vectors.key2row
    assert nlp.vocab['A'].has_vector is True
    assert nlp.vocab['a'].has_vector is True
    assert nlp('A')[0].has_vector is True
    assert nlp('a')[0].has_vector is True
    assert nlp.vocab['D'].has_vector is False
    assert nlp.vocab['d'].has_vector is False
    nlp.vocab.set_vector('D', numpy.asarray([4, 5, 6]))
    assert nlp.vocab['D'].has_vector is True
    assert nlp.vocab['d'].has_vector is True