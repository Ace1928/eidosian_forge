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
def test_example_from_dict_no_ner(en_vocab):
    words = ['a', 'b', 'c', 'd']
    spaces = [True, True, False, True]
    predicted = Doc(en_vocab, words=words, spaces=spaces)
    example = Example.from_dict(predicted, {'words': words})
    ner_tags = example.get_aligned_ner()
    assert ner_tags == [None, None, None, None]