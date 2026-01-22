import pytest
from spacy.lang.zh import Chinese
from ...util import make_tempdir
def test_zh_tokenizer_serialize_char(zh_tokenizer_char):
    zh_tokenizer_serialize(zh_tokenizer_char)