import random
import numpy
import pytest
import srsly
from thinc.api import Adam, compounding
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.training import (
from spacy.training.align import get_alignments
from spacy.training.alignment_array import AlignmentArray
from spacy.training.converters import json_to_docs
from spacy.training.loop import train_while_improving
from spacy.util import (
from ..util import make_tempdir
def test_aligned_spans_x2y(en_vocab, en_tokenizer):
    text = 'Mr and Mrs Smith flew to San Francisco Valley'
    nlp = English()
    patterns = [{'label': 'PERSON', 'pattern': 'Mr and Mrs Smith'}, {'label': 'LOC', 'pattern': 'San Francisco Valley'}]
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    doc = nlp(text)
    assert [(ent.start, ent.end) for ent in doc.ents] == [(0, 4), (6, 9)]
    prefix = 'Mr and Mrs Smith flew to '
    entities = [(0, len('Mr and Mrs Smith'), 'PERSON'), (len(prefix), len(prefix + 'San Francisco Valley'), 'LOC')]
    tokens_ref = ['Mr and Mrs', 'Smith', 'flew', 'to', 'San Francisco', 'Valley']
    example = Example.from_dict(doc, {'words': tokens_ref, 'entities': entities})
    assert [(ent.start, ent.end) for ent in example.reference.ents] == [(0, 2), (4, 6)]
    ents_pred = example.predicted.ents
    assert [(ent.start, ent.end) for ent in ents_pred] == [(0, 4), (6, 9)]
    ents_x2y = example.get_aligned_spans_x2y(ents_pred)
    assert [(ent.start, ent.end) for ent in ents_x2y] == [(0, 2), (4, 6)]