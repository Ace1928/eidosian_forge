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
def test_vectors_clear():
    data = OPS.asarray([[4, 2, 2, 2], [4, 2, 2, 2], [1, 1, 1, 1]], dtype='f')
    v = Vectors(data=data, keys=['A', 'B', 'C'])
    assert v.is_full is True
    assert hash_string('A') in v
    v.clear()
    assert v.key2row == {}
    assert list(v) == []
    assert v.is_full is False
    assert 'A' not in v
    with pytest.raises(KeyError):
        v['A']