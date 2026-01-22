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
def test_gold_biluo_BIL(en_vocab):
    words = ['I', 'flew', 'to', 'San', 'Francisco', 'Valley', '.']
    spaces = [True, True, True, True, True, False, True]
    doc = Doc(en_vocab, words=words, spaces=spaces)
    entities = [(len('I flew to '), len('I flew to San Francisco Valley'), 'LOC')]
    tags = offsets_to_biluo_tags(doc, entities)
    assert tags == ['O', 'O', 'O', 'B-LOC', 'I-LOC', 'L-LOC', 'O']