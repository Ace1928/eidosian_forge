import numpy
import pytest
import srsly
from spacy.attrs import NORM
from spacy.lang.en import English
from spacy.strings import StringStore
from spacy.tokens import Doc
from spacy.vocab import Vocab
@pytest.mark.parametrize('text1,text2', [('hello', 'bye')])
def test_pickle_string_store(text1, text2):
    stringstore = StringStore()
    store1 = stringstore[text1]
    store2 = stringstore[text2]
    data = srsly.pickle_dumps(stringstore, protocol=-1)
    unpickled = srsly.pickle_loads(data)
    assert unpickled[text1] == store1
    assert unpickled[text2] == store2
    assert len(stringstore) == len(unpickled)