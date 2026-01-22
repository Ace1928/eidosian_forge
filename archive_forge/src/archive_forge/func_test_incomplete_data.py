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
@pytest.mark.parametrize('pipe_name', ['parser', 'beam_parser'])
def test_incomplete_data(pipe_name):
    nlp = English()
    parser = nlp.add_pipe(pipe_name)
    train_examples = []
    for text, annotations in PARTIAL_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for dep in annotations.get('deps', []):
            if dep is not None:
                parser.add_label(dep)
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(150):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses[pipe_name] < 0.0001
    test_text = 'I like securities.'
    doc = nlp(test_text)
    assert doc[0].dep_ == 'nsubj'
    assert doc[2].dep_ == 'dobj'
    assert doc[0].head.i == 1
    assert doc[2].head.i == 1