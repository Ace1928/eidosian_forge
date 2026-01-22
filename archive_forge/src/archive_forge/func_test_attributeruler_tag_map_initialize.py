import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def test_attributeruler_tag_map_initialize(nlp, tag_map):
    ruler = nlp.add_pipe('attribute_ruler')
    ruler.initialize(lambda: [], tag_map=tag_map)
    check_tag_map(ruler)