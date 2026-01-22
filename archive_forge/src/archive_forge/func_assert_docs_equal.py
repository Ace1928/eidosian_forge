import contextlib
import re
import tempfile
import numpy
import srsly
from thinc.api import get_current_ops
from spacy.tokens import Doc
from spacy.training import split_bilu_label
from spacy.util import make_tempdir  # noqa: F401
from spacy.vocab import Vocab
def assert_docs_equal(doc1, doc2):
    """Compare two Doc objects and assert that they're equal. Tests for tokens,
    tags, dependencies and entities."""
    assert [t.orth for t in doc1] == [t.orth for t in doc2]
    assert [t.pos for t in doc1] == [t.pos for t in doc2]
    assert [t.tag for t in doc1] == [t.tag for t in doc2]
    assert [t.head.i for t in doc1] == [t.head.i for t in doc2]
    assert [t.dep for t in doc1] == [t.dep for t in doc2]
    assert [t.is_sent_start for t in doc1] == [t.is_sent_start for t in doc2]
    assert [t.ent_type for t in doc1] == [t.ent_type for t in doc2]
    assert [t.ent_iob for t in doc1] == [t.ent_iob for t in doc2]
    for ent1, ent2 in zip(doc1.ents, doc2.ents):
        assert ent1.start == ent2.start
        assert ent1.end == ent2.end
        assert ent1.label == ent2.label
        assert ent1.kb_id == ent2.kb_id