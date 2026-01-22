import pytest
from numpy.testing import assert_array_equal
from thinc.api import Config, get_current_ops
from spacy import util
from spacy.lang.en import English
from spacy.ml.models.tok2vec import (
from spacy.pipeline.tok2vec import Tok2Vec, Tok2VecListener
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import registry
from spacy.vocab import Vocab
from ..util import add_vecs_to_vocab, get_batch, make_tempdir
@pytest.mark.parametrize('batch_size,width,embed_size', [[1, 128, 2000], [2, 128, 2000], [3, 8, 63]])
def test_tok2vec_batch_sizes(batch_size, width, embed_size):
    batch = get_batch(batch_size)
    tok2vec = build_Tok2Vec_model(MultiHashEmbed(width=width, rows=[embed_size] * 4, include_static_vectors=False, attrs=['NORM', 'PREFIX', 'SUFFIX', 'SHAPE']), MaxoutWindowEncoder(width=width, depth=4, window_size=1, maxout_pieces=3))
    tok2vec.initialize()
    vectors, backprop = tok2vec.begin_update(batch)
    assert len(vectors) == len(batch)
    for doc_vec, doc in zip(vectors, batch):
        assert doc_vec.shape == (len(doc), width)