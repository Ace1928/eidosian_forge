import pytest
from spacy.language import Language
from spacy.pipeline.functions import merge_subtokens
from spacy.tokens import Doc, Span
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.usefixtures('clean_underscore')
def test_factories_doc_cleaner():
    nlp = Language()
    nlp.add_pipe('doc_cleaner')
    doc = nlp.make_doc('text')
    doc.tensor = [1, 2, 3]
    doc = nlp(doc)
    assert doc.tensor is None
    nlp = Language()
    nlp.add_pipe('doc_cleaner', config={'silent': False})
    with pytest.warns(UserWarning):
        doc = nlp('text')
    Doc.set_extension('test_attr', default=-1)
    nlp = Language()
    nlp.add_pipe('doc_cleaner', config={'attrs': {'_.test_attr': 0}})
    doc = nlp.make_doc('text')
    doc._.test_attr = 100
    doc = nlp(doc)
    assert doc._.test_attr == 0