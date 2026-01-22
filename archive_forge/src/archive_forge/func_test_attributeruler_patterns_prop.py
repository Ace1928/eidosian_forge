import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def test_attributeruler_patterns_prop(nlp, pattern_dicts):
    a = nlp.add_pipe('attribute_ruler')
    a.add_patterns(pattern_dicts)
    for p1, p2 in zip(pattern_dicts, a.patterns):
        assert p1['patterns'] == p2['patterns']
        assert p1['attrs'] == p2['attrs']
        if p1.get('index'):
            assert p1['index'] == p2['index']