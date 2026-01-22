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
def test_aligned_spans_y2x_overlap(en_vocab, en_tokenizer):
    text = 'I flew to San Francisco Valley'
    nlp = English()
    doc = nlp(text)
    gold_doc = nlp.make_doc(text)
    spans = []
    prefix = 'I flew to '
    spans.append(gold_doc.char_span(len(prefix), len(prefix + 'San Francisco'), label='CITY'))
    spans.append(gold_doc.char_span(len(prefix), len(prefix + 'San Francisco Valley'), label='VALLEY'))
    spans_key = 'overlap_ents'
    gold_doc.spans[spans_key] = spans
    example = Example(doc, gold_doc)
    spans_gold = example.reference.spans[spans_key]
    assert [(ent.start, ent.end) for ent in spans_gold] == [(3, 5), (3, 6)]
    spans_y2x_no_overlap = example.get_aligned_spans_y2x(spans_gold, allow_overlap=False)
    assert [(ent.start, ent.end) for ent in spans_y2x_no_overlap] == [(3, 5)]
    spans_y2x_overlap = example.get_aligned_spans_y2x(spans_gold, allow_overlap=True)
    assert [(ent.start, ent.end) for ent in spans_y2x_overlap] == [(3, 5), (3, 6)]