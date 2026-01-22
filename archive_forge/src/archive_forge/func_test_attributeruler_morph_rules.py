import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def test_attributeruler_morph_rules(nlp, morph_rules):
    ruler = AttributeRuler(nlp.vocab)
    ruler.load_from_morph_rules(morph_rules)
    check_morph_rules(ruler)