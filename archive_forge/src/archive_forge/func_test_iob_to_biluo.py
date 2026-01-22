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
def test_iob_to_biluo():
    good_iob = ['O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-PERSON']
    good_biluo = ['O', 'O', 'B-LOC', 'L-LOC', 'O', 'U-PERSON']
    bad_iob = ['O', 'O', '"', 'B-LOC', 'I-LOC']
    converted_biluo = iob_to_biluo(good_iob)
    assert good_biluo == converted_biluo
    with pytest.raises(ValueError):
        iob_to_biluo(bad_iob)