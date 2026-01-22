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
def test_init_vectors_unset():
    v = Vectors(shape=(10, 10))
    assert v.is_full is False
    assert v.shape == (10, 10)
    with pytest.raises(ValueError):
        v = Vectors(shape=(10, 10), mode='floret')
    v = Vectors(data=OPS.xp.zeros((10, 10)), mode='floret', hash_count=1)
    assert v.is_full is True