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
def test_gold_biluo_additional_whitespace(en_vocab, en_tokenizer):
    words, spaces = get_words_and_spaces(['I', 'flew', 'to', 'San Francisco', 'Valley', '.'], 'I flew  to San Francisco Valley.')
    doc = Doc(en_vocab, words=words, spaces=spaces)
    prefix = 'I flew  to '
    entities = [(len(prefix), len(prefix + 'San Francisco Valley'), 'LOC')]
    gold_words = ['I', 'flew', ' ', 'to', 'San Francisco Valley', '.']
    gold_spaces = [True, True, False, True, False, False]
    example = Example.from_dict(doc, {'words': gold_words, 'spaces': gold_spaces, 'entities': entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ['O', 'O', 'O', 'O', 'B-LOC', 'L-LOC', 'O']