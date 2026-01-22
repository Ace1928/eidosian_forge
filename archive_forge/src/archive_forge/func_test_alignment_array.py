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
def test_alignment_array():
    a = AlignmentArray([[0, 1, 2], [3], [], [4, 5, 6, 7], [8, 9]])
    assert list(a.data) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert list(a.lengths) == [3, 1, 0, 4, 2]
    assert list(a[3]) == [4, 5, 6, 7]
    assert list(a[2]) == []
    assert list(a[-2]) == [4, 5, 6, 7]
    assert list(a[1:4]) == [3, 4, 5, 6, 7]
    assert list(a[1:]) == [3, 4, 5, 6, 7, 8, 9]
    assert list(a[:3]) == [0, 1, 2, 3]
    assert list(a[:]) == list(a.data)
    assert list(a[0:0]) == []
    assert list(a[3:3]) == []
    assert list(a[-1:-1]) == []
    with pytest.raises(ValueError, match='only supports slicing with a step of 1'):
        a[:4:-1]
    with pytest.raises(ValueError, match='only supports indexing using an int or a slice'):
        a[[0, 1, 3]]
    a = AlignmentArray([[], [1, 2, 3], [4, 5]])
    assert list(a[0]) == []
    assert list(a[0:1]) == []
    assert list(a[2]) == [4, 5]
    assert list(a[0:2]) == [1, 2, 3]
    a = AlignmentArray([[1, 2, 3], [4, 5], []])
    assert list(a[-1]) == []
    assert list(a[-2:]) == [4, 5]