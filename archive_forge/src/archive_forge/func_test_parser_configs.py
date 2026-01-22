import pytest
from numpy.testing import assert_equal
from thinc.api import Adam
from spacy import registry, util
from spacy.attrs import DEP, NORM
from spacy.lang.en import English
from spacy.pipeline import DependencyParser
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
from ..util import apply_transition_sequence, make_tempdir
@pytest.mark.slow
@pytest.mark.parametrize('pipe_name', ['parser', 'beam_parser'])
@pytest.mark.parametrize('parser_config', [{'@architectures': 'spacy.TransitionBasedParser.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'state_type': 'parser', 'extra_state_tokens': False, 'hidden_width': 64, 'maxout_pieces': 2, 'use_upper': True}, {'@architectures': 'spacy.TransitionBasedParser.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'state_type': 'parser', 'extra_state_tokens': False, 'hidden_width': 64, 'maxout_pieces': 2, 'use_upper': True}])
def test_parser_configs(pipe_name, parser_config):
    pipe_config = {'model': parser_config}
    nlp = English()
    parser = nlp.add_pipe(pipe_name, config=pipe_config)
    train_examples = []
    for text, annotations in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for dep in annotations.get('deps', []):
            parser.add_label(dep)
    optimizer = nlp.initialize()
    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)