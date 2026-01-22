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
@pytest.mark.parametrize('with_vectors', (False, True))
def test_tok2vec_listener(with_vectors):
    orig_config = Config().from_str(cfg_string)
    orig_config['components']['tok2vec']['model']['embed']['include_static_vectors'] = with_vectors
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    if with_vectors:
        ops = get_current_ops()
        vectors = [('apple', ops.asarray([1, 2, 3])), ('orange', ops.asarray([-1, -2, -3])), ('and', ops.asarray([-1, -1, -1])), ('juice', ops.asarray([5, 5, 10])), ('pie', ops.asarray([7, 6.3, 8.9]))]
        add_vecs_to_vocab(nlp.vocab, vectors)
    assert nlp.pipe_names == ['tok2vec', 'tagger']
    tagger = nlp.get_pipe('tagger')
    tok2vec = nlp.get_pipe('tok2vec')
    tagger_tok2vec = tagger.model.get_ref('tok2vec')
    assert isinstance(tok2vec, Tok2Vec)
    assert isinstance(tagger_tok2vec, Tok2VecListener)
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
        for tag in t[1]['tags']:
            tagger.add_label(tag)
    optimizer = nlp.initialize(lambda: train_examples)
    assert tok2vec.listeners == [tagger_tok2vec]
    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    doc = nlp('Running the pipeline as a whole.')
    doc_tensor = tagger_tok2vec.predict([doc])[0]
    ops = get_current_ops()
    assert_array_equal(ops.to_numpy(doc.tensor), ops.to_numpy(doc_tensor))
    doc = nlp('')
    nlp.select_pipes(disable='tok2vec')
    assert nlp.pipe_names == ['tagger']
    nlp('Running the pipeline with the Tok2Vec component disabled.')