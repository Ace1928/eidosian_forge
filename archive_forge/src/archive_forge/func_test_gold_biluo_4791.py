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
def test_gold_biluo_4791(en_vocab, en_tokenizer):
    doc = en_tokenizer("I'll return the A54 amount")
    gold_words = ['I', "'ll", 'return', 'the', 'A', '54', 'amount']
    gold_spaces = [False, True, True, True, False, True, False]
    entities = [(16, 19, 'MONEY')]
    example = Example.from_dict(doc, {'words': gold_words, 'spaces': gold_spaces, 'entities': entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ['O', 'O', 'O', 'O', 'U-MONEY', 'O']
    doc = en_tokenizer("I'll return the $54 amount")
    gold_words = ['I', "'ll", 'return', 'the', '$', '54', 'amount']
    gold_spaces = [False, True, True, True, False, True, False]
    entities = [(16, 19, 'MONEY')]
    example = Example.from_dict(doc, {'words': gold_words, 'spaces': gold_spaces, 'entities': entities})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == ['O', 'O', 'O', 'O', 'B-MONEY', 'L-MONEY', 'O']