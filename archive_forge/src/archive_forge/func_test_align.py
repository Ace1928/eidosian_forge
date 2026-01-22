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
@pytest.mark.parametrize('tokens_a,tokens_b,expected', [(['a', 'b', 'c'], ['ab', 'c'], ([[0], [0], [1]], [[0, 1], [2]])), (['a', 'b', '"', 'c'], ['ab"', 'c'], ([[0], [0], [0], [1]], [[0, 1, 2], [3]])), (['a', 'bc'], ['ab', 'c'], ([[0], [0, 1]], [[0, 1], [1]])), (['ab', 'c', 'd'], ['a', 'b', 'cd'], ([[0, 1], [2], [2]], [[0], [0], [1, 2]])), (['a', 'b', 'cd'], ['a', 'b', 'c', 'd'], ([[0], [1], [2, 3]], [[0], [1], [2], [2]])), ([' ', 'a'], ['a'], ([[], [0]], [[1]])), (['a', "''", "'", ','], ["a'", "''", ','], ([[0], [0, 1], [1], [2]], [[0, 1], [1, 2], [3]]))])
def test_align(tokens_a, tokens_b, expected):
    a2b, b2a = get_alignments(tokens_a, tokens_b)
    assert (a2b, b2a) == expected
    a2b, b2a = get_alignments(tokens_b, tokens_a)
    assert (b2a, a2b) == expected