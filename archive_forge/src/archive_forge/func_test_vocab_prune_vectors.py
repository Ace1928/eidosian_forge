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
def test_vocab_prune_vectors():
    vocab = Vocab(vectors_name='test_vocab_prune_vectors')
    _ = vocab['cat']
    _ = vocab['dog']
    _ = vocab['kitten']
    data = OPS.xp.ndarray((5, 3), dtype='f')
    data[0] = OPS.asarray([1.0, 1.2, 1.1])
    data[1] = OPS.asarray([0.3, 1.3, 1.0])
    data[2] = OPS.asarray([0.9, 1.22, 1.05])
    vocab.set_vector('cat', data[0])
    vocab.set_vector('dog', data[1])
    vocab.set_vector('kitten', data[2])
    remap = vocab.prune_vectors(2, batch_size=2)
    assert list(remap.keys()) == ['kitten']
    neighbour, similarity = list(remap.values())[0]
    assert neighbour == 'cat', remap
    cosine = get_cosine(data[0], data[2])
    assert_allclose(float(similarity), cosine, atol=0.0001, rtol=0.001)