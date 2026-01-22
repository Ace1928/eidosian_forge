import pytest
from spacy.lookups import Lookups, Table
from spacy.strings import get_string_id
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_lookups_to_from_disk_via_vocab():
    table_name = 'test'
    vocab = Vocab()
    vocab.lookups.add_table(table_name, {'foo': 'bar', 'hello': 'world'})
    assert table_name in vocab.lookups
    with make_tempdir() as tmpdir:
        vocab.to_disk(tmpdir)
        new_vocab = Vocab()
        new_vocab.from_disk(tmpdir)
    assert len(new_vocab.lookups) == len(vocab.lookups)
    assert table_name in new_vocab.lookups
    table = new_vocab.lookups.get_table(table_name)
    assert len(table) == 2
    assert table['hello'] == 'world'