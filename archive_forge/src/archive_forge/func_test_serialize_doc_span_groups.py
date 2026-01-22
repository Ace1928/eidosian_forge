import copy
import pickle
import numpy
import pytest
from spacy.attrs import DEP, HEAD
from spacy.lang.en import English
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_serialize_doc_span_groups(en_vocab):
    doc = Doc(en_vocab, words=['hello', 'world', '!'])
    span = doc[0:2]
    span.label_ = 'test_serialize_doc_span_groups_label'
    span.id_ = 'test_serialize_doc_span_groups_id'
    span.kb_id_ = 'test_serialize_doc_span_groups_kb_id'
    doc.spans['content'] = [span]
    new_doc = Doc(en_vocab).from_bytes(doc.to_bytes())
    assert len(new_doc.spans['content']) == 1
    assert new_doc.spans['content'][0].label_ == 'test_serialize_doc_span_groups_label'
    assert new_doc.spans['content'][0].id_ == 'test_serialize_doc_span_groups_id'
    assert new_doc.spans['content'][0].kb_id_ == 'test_serialize_doc_span_groups_kb_id'