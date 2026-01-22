import pytest
from spacy.lookups import Lookups, Table
from spacy.strings import get_string_id
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_lookups_to_from_bytes_via_vocab():
    table_name = 'test'
    vocab = Vocab()
    vocab.lookups.add_table(table_name, {'foo': 'bar', 'hello': 'world'})
    assert table_name in vocab.lookups
    vocab_bytes = vocab.to_bytes()
    new_vocab = Vocab()
    new_vocab.from_bytes(vocab_bytes)
    assert len(new_vocab.lookups) == len(vocab.lookups)
    assert table_name in new_vocab.lookups
    table = new_vocab.lookups.get_table(table_name)
    assert len(table) == 2
    assert table['hello'] == 'world'
    assert new_vocab.to_bytes() == vocab_bytes