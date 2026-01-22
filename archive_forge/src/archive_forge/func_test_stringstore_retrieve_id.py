import pytest
from spacy.strings import StringStore
@pytest.mark.parametrize('text', [b'A'])
def test_stringstore_retrieve_id(stringstore, text):
    key = stringstore.add(text)
    assert len(stringstore) == 1
    assert stringstore[key] == text.decode('utf8')
    with pytest.raises(KeyError):
        stringstore[20000]