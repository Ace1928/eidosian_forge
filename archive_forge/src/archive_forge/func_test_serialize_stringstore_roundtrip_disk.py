import pickle
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.strings import StringStore
from spacy.tokens import Doc
from spacy.util import ensure_path, load_model
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.parametrize('strings1,strings2', test_strings)
def test_serialize_stringstore_roundtrip_disk(strings1, strings2):
    sstore1 = StringStore(strings=strings1)
    sstore2 = StringStore(strings=strings2)
    with make_tempdir() as d:
        file_path1 = d / 'strings1'
        file_path2 = d / 'strings2'
        sstore1.to_disk(file_path1)
        sstore2.to_disk(file_path2)
        sstore1_d = StringStore().from_disk(file_path1)
        sstore2_d = StringStore().from_disk(file_path2)
        assert set(sstore1_d) == set(sstore1)
        assert set(sstore2_d) == set(sstore2)
        if set(strings1) == set(strings2):
            assert set(sstore1_d) == set(sstore2_d)
        else:
            assert set(sstore1_d) != set(sstore2_d)