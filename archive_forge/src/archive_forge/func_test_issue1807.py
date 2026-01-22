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
@pytest.mark.issue(1807)
def test_issue1807():
    """Test vocab.set_vector also adds the word to the vocab."""
    vocab = Vocab(vectors_name='test_issue1807')
    assert 'hello' not in vocab
    vocab.set_vector('hello', numpy.ones((50,), dtype='f'))
    assert 'hello' in vocab