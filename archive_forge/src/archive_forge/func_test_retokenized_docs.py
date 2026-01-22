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
def test_retokenized_docs(doc):
    a = doc.to_array(['TAG'])
    doc1 = Doc(doc.vocab, words=[t.text for t in doc]).from_array(['TAG'], a)
    doc2 = Doc(doc.vocab, words=[t.text for t in doc]).from_array(['TAG'], a)
    example = Example(doc1, doc2)
    expected1 = ['Sarah', "'s", 'sister', 'flew', 'to', 'Silicon', 'Valley', 'via', 'London', '.']
    expected2 = [None, 'sister', 'flew', 'to', None, 'via', 'London', '.']
    assert example.get_aligned('ORTH', as_string=True) == expected1
    with doc1.retokenize() as retokenizer:
        retokenizer.merge(doc1[0:2])
        retokenizer.merge(doc1[5:7])
    assert example.get_aligned('ORTH', as_string=True) == expected2