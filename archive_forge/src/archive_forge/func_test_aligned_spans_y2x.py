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
def test_aligned_spans_y2x(en_vocab, en_tokenizer):
    words = ['Mr and Mrs Smith', 'flew', 'to', 'San Francisco Valley', '.']
    spaces = [True, True, True, False, False]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    prefix = 'Mr and Mrs Smith flew to '
    entities = [(0, len('Mr and Mrs Smith'), 'PERSON'), (len(prefix), len(prefix + 'San Francisco Valley'), 'LOC')]
    tokens_ref = ['Mr', 'and', 'Mrs', 'Smith', 'flew', 'to', 'San', 'Francisco', 'Valley', '.']
    example = Example.from_dict(doc, {'words': tokens_ref, 'entities': entities})
    ents_ref = example.reference.ents
    assert [(ent.start, ent.end) for ent in ents_ref] == [(0, 4), (6, 9)]
    ents_y2x = example.get_aligned_spans_y2x(ents_ref)
    assert [(ent.start, ent.end) for ent in ents_y2x] == [(0, 1), (3, 4)]