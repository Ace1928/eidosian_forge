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
def test_get_vector_resize(strings, data):
    strings = [hash_string(s) for s in strings]
    v = Vectors(data=data)
    resized_dim = v.shape[1] - 1
    v.resize(shape=(v.shape[0], resized_dim))
    for i, string in enumerate(strings):
        v.add(string, row=i)
    assert list(v[strings[0]]) == list(data[0, :resized_dim])
    assert list(v[strings[1]]) == list(data[1, :resized_dim])
    v = Vectors(data=data)
    resized_dim = v.shape[1] + 1
    v.resize(shape=(v.shape[0], resized_dim))
    for i, string in enumerate(strings):
        v.add(string, row=i)
    assert list(v[strings[0]]) == list(data[0]) + [0]
    assert list(v[strings[1]]) == list(data[1]) + [0]