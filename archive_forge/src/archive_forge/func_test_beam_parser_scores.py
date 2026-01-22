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
def test_beam_parser_scores():
    beam_width = 16
    beam_density = 0.0001
    nlp = English()
    config = {'beam_width': beam_width, 'beam_density': beam_density}
    parser = nlp.add_pipe('beam_parser', config=config)
    train_examples = []
    for text, annotations in CONFLICTING_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for dep in annotations.get('deps', []):
            parser.add_label(dep)
    optimizer = nlp.initialize()
    for i in range(10):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    test_text = 'I like securities.'
    doc = nlp.make_doc(test_text)
    docs = [doc]
    beams = parser.predict(docs)
    head_scores, label_scores = parser.scored_parses(beams)
    for j in range(len(doc)):
        for label in parser.labels:
            label_score = label_scores[0][j, label]
            assert 0 - eps <= label_score <= 1 + eps
        for i in range(len(doc)):
            head_score = head_scores[0][j, i]
            assert 0 - eps <= head_score <= 1 + eps