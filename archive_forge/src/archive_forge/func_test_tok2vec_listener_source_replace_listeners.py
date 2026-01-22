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
def test_tok2vec_listener_source_replace_listeners():
    orig_config = Config().from_str(cfg_string_multi)
    nlp1 = util.load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp1.get_pipe('tok2vec').listening_components == ['tagger', 'ner']
    nlp1.replace_listeners('tok2vec', 'tagger', ['model.tok2vec'])
    assert nlp1.get_pipe('tok2vec').listening_components == ['ner']
    nlp2 = English()
    nlp2.add_pipe('tok2vec', source=nlp1)
    assert nlp2.get_pipe('tok2vec').listening_components == []
    nlp2.add_pipe('tagger', source=nlp1)
    assert nlp2.get_pipe('tok2vec').listening_components == []
    nlp2.add_pipe('ner', name='ner2', source=nlp1)
    assert nlp2.get_pipe('tok2vec').listening_components == ['ner2']