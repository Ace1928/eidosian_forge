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
def test_gold_ner_missing_tags(en_tokenizer):
    doc = en_tokenizer('I flew to Silicon Valley via London.')
    biluo_tags = [None, 'O', 'O', 'B-LOC', 'L-LOC', 'O', 'U-GPE', 'O']
    example = Example.from_dict(doc, {'entities': biluo_tags})
    assert example.get_aligned('ENT_IOB') == [0, 2, 2, 3, 1, 2, 3, 2]