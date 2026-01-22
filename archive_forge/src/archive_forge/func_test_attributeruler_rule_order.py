import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def test_attributeruler_rule_order(nlp):
    a = AttributeRuler(nlp.vocab)
    patterns = [{'patterns': [[{'TAG': 'VBZ'}]], 'attrs': {'POS': 'VERB'}}, {'patterns': [[{'TAG': 'VBZ'}]], 'attrs': {'POS': 'NOUN'}}]
    a.add_patterns(patterns)
    doc = Doc(nlp.vocab, words=['This', 'is', 'a', 'test', '.'], tags=['DT', 'VBZ', 'DT', 'NN', '.'])
    doc = a(doc)
    assert doc[1].pos_ == 'NOUN'