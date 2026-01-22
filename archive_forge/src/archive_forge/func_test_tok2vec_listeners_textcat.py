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
def test_tok2vec_listeners_textcat():
    orig_config = Config().from_str(cfg_string_multi_textcat)
    nlp = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.pipe_names == ['tok2vec', 'textcat_multilabel', 'tagger']
    tagger = nlp.get_pipe('tagger')
    textcat = nlp.get_pipe('textcat_multilabel')
    tok2vec = nlp.get_pipe('tok2vec')
    tagger_tok2vec = tagger.model.get_ref('tok2vec')
    textcat_tok2vec = textcat.model.get_ref('tok2vec')
    assert isinstance(tok2vec, Tok2Vec)
    assert isinstance(tagger_tok2vec, Tok2VecListener)
    assert isinstance(textcat_tok2vec, Tok2VecListener)
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    optimizer = nlp.initialize(lambda: train_examples)
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    docs = list(nlp.pipe(['Eat blue ham', 'I like green eggs']))
    cats0 = docs[0].cats
    assert cats0['preference'] < 0.1
    assert cats0['imperative'] > 0.9
    cats1 = docs[1].cats
    assert cats1['preference'] > 0.1
    assert cats1['imperative'] < 0.9
    assert [t.tag_ for t in docs[0]] == ['V', 'J', 'N']
    assert [t.tag_ for t in docs[1]] == ['N', 'V', 'J', 'N']