import pytest
from thinc.api import Model
from spacy import registry
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.pipeline.transition_parser import Parser
from spacy.tokens.doc import Doc
from spacy.training import Example
from spacy.vocab import Vocab
@pytest.fixture
def tok2vec():
    cfg = {'model': DEFAULT_TOK2VEC_MODEL}
    tok2vec = registry.resolve(cfg, validate=True)['model']
    tok2vec.initialize()
    return tok2vec