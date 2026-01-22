import pytest
from thinc.api import Adam, fix_random_seed
from spacy import registry
from spacy.attrs import NORM
from spacy.language import Language
from spacy.pipeline import DependencyParser, EntityRecognizer
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_ner_labels_added_implicitly_on_greedy_parse():
    nlp = Language()
    ner = nlp.add_pipe('beam_ner')
    for label in ['A', 'B', 'C']:
        ner.add_label(label)
    nlp.initialize()
    doc = Doc(nlp.vocab, words=['hello', 'world'], ents=['B-D', 'O'])
    ner.greedy_parse([doc])
    assert 'D' in ner.labels