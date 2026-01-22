import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
def test_attributeruler_serialize(nlp, pattern_dicts):
    a = nlp.add_pipe('attribute_ruler')
    a.add_patterns(pattern_dicts)
    text = 'This is a test.'
    attrs = ['ORTH', 'LEMMA', 'MORPH']
    doc = nlp(text)
    a_reloaded = AttributeRuler(nlp.vocab).from_bytes(a.to_bytes())
    assert a.to_bytes() == a_reloaded.to_bytes()
    doc1 = a_reloaded(nlp.make_doc(text))
    numpy.array_equal(doc.to_array(attrs), doc1.to_array(attrs))
    assert a.patterns == a_reloaded.patterns
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(text)
        assert nlp2.get_pipe('attribute_ruler').to_bytes() == a.to_bytes()
        assert numpy.array_equal(doc.to_array(attrs), doc2.to_array(attrs))
        assert a.patterns == nlp2.get_pipe('attribute_ruler').patterns